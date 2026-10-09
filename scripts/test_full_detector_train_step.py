
import torch
import torch.nn.functional as F

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.full_detector import (
    FullBEVHeightDetector,
)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda")

    assert torch.cuda.is_available()

    # --------------------------------------------------
    # 1. 加载真实 S3S 数据
    # --------------------------------------------------
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    samples = [dataset[0], dataset[1]]

    images = torch.stack([
        sample["image"] for sample in samples
    ]).to(device)

    K = torch.stack([
        torch.as_tensor(
            sample["intrinsics"],
            dtype=torch.float32,
        )
        for sample in samples
    ]).to(device)

    T = torch.stack([
        torch.as_tensor(
            sample["extrinsics_raw"],
            dtype=torch.float32,
        )
        for sample in samples
    ]).to(device)

    boxes_list = [
        sample["boxes_raw"].to(device)
        for sample in samples
    ]

    print(
        "帧号:",
        [sample["frame_id"] for sample in samples]
    )

    print(
        "真实车辆数量:",
        [len(boxes) for boxes in boxes_list]
    )

    assert sum(len(b) for b in boxes_list) > 0

    # --------------------------------------------------
    # 2. 缩小训练测试分辨率并同步缩放内参
    # --------------------------------------------------
    images = F.interpolate(
        images,
        size=(288, 512),
        mode="bilinear",
        align_corners=False,
    )

    K = K.clone()
    K[:, 0, :] *= 0.5
    K[:, 1, :] *= 0.5

    print("输入RGB:", tuple(images.shape))
    print("内参:", tuple(K.shape))
    print("外参:", tuple(T.shape))

    # --------------------------------------------------
    # 3. 构建完整网络
    # --------------------------------------------------
    torch.cuda.reset_peak_memory_stats(device)

    model = FullBEVHeightDetector(
        backbone_pretrained=True,
        backbone_depth=50,
        ground_z=-30.0,
    ).to(device)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    param_count = sum(
        p.numel() for p in model.parameters()
    )

    print("完整网络参数量:", f"{param_count:,}")

    # --------------------------------------------------
    # 4. 正式检测前向传播
    # --------------------------------------------------
    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        images=images,
        K=K,
        world2cam=T,
    )

    predictions = outputs["predictions"]

    print("\n网络输出:")
    print("BEV:", tuple(outputs["bev"].shape))
    print(
        "BEV Features:",
        tuple(outputs["bev_features"].shape)
    )

    for name, value in predictions.items():
        print(name, tuple(value.shape))
        assert torch.isfinite(value).all()

    assert outputs["bev"].shape == (
        2, 80, 80, 128
    )

    assert outputs["bev_features"].shape == (
        2, 256, 80, 128
    )

    assert predictions["heatmap"].shape == (
        2, 1, 80, 128
    )

    # --------------------------------------------------
    # 5. 利用真实车辆框计算正式检测损失
    # --------------------------------------------------
    losses, targets = model.compute_loss(
        predictions,
        boxes_list,
    )

    num_targets = int(
        targets["mask"].sum().item()
    )

    expected_targets = sum(
        len(boxes) for boxes in boxes_list
    )

    print("\n真实车辆监督:")
    print("GT数量:", expected_targets)
    print("有效目标:", num_targets)

    assert num_targets == expected_targets

    print("\n检测损失:")

    for name, value in losses.items():
        print(
            f"{name}: {value.item():.6f}"
        )
        assert torch.isfinite(value)

    total_loss = losses["total"]

    assert total_loss.item() > 0

    # --------------------------------------------------
    # 6. 从真实检测损失反向传播
    # --------------------------------------------------
    total_loss.backward()

    checks = {
        "ResNet50":
            model.bev_extractor.image_encoder
            .backbone.stem[0].weight,

        "Image SECOND-FPN":
            model.bev_extractor.image_encoder
            .backbone.fpn.deblocks[0][0].weight,

        "HeightNet Context":
            model.bev_extractor.image_encoder
            .heightnet.context_conv.weight,

        "HeightNet Height":
            model.bev_extractor.image_encoder
            .heightnet.height_layer.weight,

        "HeightNet DCN":
            model.bev_extractor.image_encoder
            .heightnet.height_conv[-1].weight,

        "BEV Encoder":
            model.bev_encoder.layer1[0].conv1.weight,

        "BEV SECOND-FPN":
            model.bev_encoder
            .fpn.deblocks[0][0].weight,

        "CenterPoint Heatmap":
            model.detection_head.task_head
            .branches["heatmap"][-1].weight,

        "CenterPoint Reg":
            model.detection_head.task_head
            .branches["reg"][-1].weight,
    }

    print("\n真实检测损失梯度检查:")

    for name, parameter in checks.items():
        grad = parameter.grad

        ok = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}: {ok}")

        assert ok, f"{name}梯度异常"

    # --------------------------------------------------
    # 7. 确认优化器真正更新了参数
    # --------------------------------------------------
    before = {
        name: parameter.detach().clone()
        for name, parameter in checks.items()
    }

    optimizer.step()

    print("\n参数更新检查:")

    for name, parameter in checks.items():
        changed = not torch.equal(
            before[name],
            parameter.detach(),
        )

        finite = torch.isfinite(
            parameter.detach()
        ).all().item()

        print(
            f"{name}: changed={changed}, finite={finite}"
        )

        assert changed, f"{name}参数未更新"
        assert finite, f"{name}更新后出现无效值"

    torch.cuda.synchronize()

    peak_mb = (
        torch.cuda.max_memory_allocated(device)
        / 1024**2
    )

    print(
        "\nGPU峰值分配显存(MB):",
        round(peak_mb, 2),
    )

    print(
        "\n完整高规格 BEVHeight "
        "真实数据单步训练通过"
    )


if __name__ == "__main__":
    main()
