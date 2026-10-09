
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.targets import BEVTargetEncoder
from src.heightnet.roadside_detector import RoadsideVehicleDetector
from src.heightnet.losses import BEVDetectionLoss


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 使用本次覆盖率更好的图像分辨率
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    # 选择已经完成几何验证的第257帧
    index = next(
        i for i, (fid, _, _) in enumerate(dataset.samples)
        if str(fid) == "257"
    )

    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(device)
    K = sample["intrinsics"].unsqueeze(0).to(device)
    T = sample["extrinsics_raw"].unsqueeze(0).to(device)

    # 真实车辆标签
    target_encoder = BEVTargetEncoder(
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0,
    )

    target = target_encoder(sample["boxes_raw"])

    # 完整模型与检测损失
    model = RoadsideVehicleDetector().to(device)
    criterion = BEVDetectionLoss().to(device)

    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # 1. 真实 RGB 图像 -> 检测预测
    predictions, debug = model(
        images=image,
        K=K,
        T=T,
        ground_z=-30.0,
        return_debug=True,
    )

    print("帧号:", sample["frame_id"])
    print("真实车辆数量:", len(sample["boxes_raw"]))
    print("成功编码数量:", len(target["indices"]))

    print("\n中间特征形状:")
    print("RGB:", image.shape)
    print("Context:", debug["context"].shape)
    print("Height:", debug["height_prob"].shape)
    print("BEV:", debug["bev"].shape)

    print("\n检测头输出:")
    for name, value in predictions.items():
        print(name, value.shape)

    # 2. 检查 BEV 与标签尺寸一致
    assert debug["bev"].shape == (1, 80, 80, 128)
    assert predictions["heatmap_logits"].shape == (
        1, 1, 80, 128
    )
    assert target["heatmap"].shape == (1, 80, 128)

    # 3. 使用真实3D Box监督计算损失
    losses = criterion(
        predictions,
        [target],
    )

    print("\n检测损失:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")
        assert torch.isfinite(value).all()

    # 4. 验证梯度能否从检测损失回到图像主干
    losses["total"].backward()

    checks = {
        "Backbone": (
            model.roadside.backbone.stem[0].weight
        ),
        "Height Head": (
            model.roadside.heightnet.height_head.weight
        ),
        "ASPP": (
            model.roadside.heightnet.height_aspp
            .project[0].weight
        ),
        "BEV Detection Head": (
            model.detector.heatmap.weight
        ),
    }

    print("\n端到端梯度检查:")

    for name, parameter in checks.items():
        grad = parameter.grad

        passed = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}: {passed}")
        assert passed, f"{name} 梯度检查失败"

    if device.type == "cuda":
        peak_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
        print(f"\nGPU峰值分配显存: {peak_mb:.2f} MB")

    print("\n端到端检测计算测试通过")


if __name__ == "__main__":
    main()
