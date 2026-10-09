
import torch
import torch.nn.functional as F

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.full_bev_extractor import (
    FullBEVFeatureExtractor,
)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda")

    # 从现有真实 S3S 数据中取两帧
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    samples = [dataset[0], dataset[1]]

    images = torch.stack(
        [s["image"] for s in samples]
    ).to(device)

    K = torch.stack([
        torch.as_tensor(
            s["intrinsics"],
            dtype=torch.float32,
        )
        for s in samples
    ]).to(device)

    T = torch.stack([
        torch.as_tensor(
            s["extrinsics_raw"],
            dtype=torch.float32,
        )
        for s in samples
    ]).to(device)

    print("原始RGB:", tuple(images.shape))
    print("相机内参:", tuple(K.shape))
    print("相机外参:", tuple(T.shape))

    # 首次端到端反向测试使用半分辨率。
    # 图像缩放时，内参必须同步缩放。
    images = F.interpolate(
        images,
        size=(288, 512),
        mode="bilinear",
        align_corners=False,
    )

    K = K.clone()
    K[:, 0, :] *= 0.5
    K[:, 1, :] *= 0.5

    print("测试RGB:", tuple(images.shape))

    torch.cuda.reset_peak_memory_stats(device)

    model = FullBEVFeatureExtractor(
        backbone_depth=50,
        backbone_pretrained=True,
        height_channels=90,
        context_channels=80,
    ).to(device)

    model.train()
    model.zero_grad(set_to_none=True)

    # 完整前向传播
    outputs = model(
        images=images,
        K=K,
        world2cam=T,
        ground_z=-30.0,
    )

    bev = outputs["bev"]
    height_prob = outputs["height_prob"]
    valid = outputs["geometry_valid"]

    print("\n完整前向结果:")
    print("BEV:", tuple(bev.shape))
    print("Height:", tuple(height_prob.shape))
    print("Geometry Valid:", tuple(valid.shape))

    assert bev.shape == (2, 80, 80, 128)
    assert height_prob.shape == (2, 90, 18, 32)
    assert valid.shape == (2, 90, 18, 32)

    assert torch.isfinite(bev).all()
    assert torch.isfinite(height_prob).all()

    assert torch.allclose(
        height_prob.sum(dim=1),
        torch.ones_like(height_prob[:, 0]),
        atol=1e-5,
    )

    nonzero = (
        bev.abs().sum(dim=1) > 1e-6
    ).sum().item()

    print("非空BEV网格总数:", nonzero)

    assert nonzero > 0

    # 使用 BEV 输出构造测试损失。
    # 这不是正式车辆检测损失。
    loss = bev.square().mean()

    print("测试损失:", loss.item())

    assert torch.isfinite(loss)
    assert loss.item() > 0

    # 检验 BEV -> HeightNet -> Backbone 的反向链路
    loss.backward()

    checks = {
        "ResNet50":
            model.image_encoder.backbone.stem[0].weight,

        "SECOND-FPN":
            model.image_encoder.backbone.fpn
            .deblocks[0][0].weight,

        "HeightNet Reduce":
            model.image_encoder.heightnet
            .reduce_conv[0].weight,

        "Context Branch":
            model.image_encoder.heightnet
            .context_conv.weight,

        "Height Branch":
            model.image_encoder.heightnet
            .height_layer.weight,

        "DCN":
            model.image_encoder.heightnet
            .height_conv[-1].weight,
    }

    print("\n梯度检查:")

    for name, parameter in checks.items():
        grad = parameter.grad

        ok = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}: {ok}")
        assert ok, f"{name}梯度异常"

    param_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\n总参数量:",
        f"{param_count:,}",
    )

    print(
        "GPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated(device)
            / 1024**2,
            2,
        ),
    )

    print(
        "\n完整RGB -> HeightNet -> BEV "
        "端到端测试通过"
    )


if __name__ == "__main__":
    main()
