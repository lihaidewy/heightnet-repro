
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.targets import BEVTargetEncoder
from src.heightnet.detector import BEVDetectionHead
from src.heightnet.losses import BEVDetectionLoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 读取第257帧的真实3D Box标注
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(288, 512)
    )

    index = next(
        i for i, (fid, _, _) in enumerate(dataset.samples)
        if fid == "257"
    )

    sample = dataset[index]

    # 2. 生成真实BEV检测标签
    encoder = BEVTargetEncoder(
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0
    )

    target = encoder(sample["boxes_raw"])

    print("帧号:", sample["frame_id"])
    print("真实车辆数量:", len(sample["boxes_raw"]))
    print("已编码车辆数量:", len(target["indices"]))
    print("Heatmap:", target["heatmap"].shape)

    assert len(target["indices"]) == len(sample["boxes_raw"])

    # 3. 随机BEV特征，用于结构测试
    bev = torch.randn(
        1, 80, 80, 128,
        device=device,
        requires_grad=True
    )

    head = BEVDetectionHead().to(device)
    criterion = BEVDetectionLoss().to(device)

    predictions = head(bev)

    print("\n检测头输出:")

    expected_shapes = {
        "heatmap_logits": (1, 1, 80, 128),
        "offset": (1, 2, 80, 128),
        "log_sizes": (1, 3, 80, 128),
        "yaws": (1, 2, 80, 128),
    }

    for name, tensor in predictions.items():
        print(name, tensor.shape)
        assert tensor.shape == expected_shapes[name]

    # 4. 计算检测损失
    losses = criterion(
        predictions,
        [target]
    )

    print("\n检测损失:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")
        assert torch.isfinite(value).all()

    # 5. 反向传播
    losses["total"].backward()

    print("\n梯度检查:")
    print("BEV输入梯度:", bev.grad is not None)

    for name in [
        "heatmap",
        "offset",
        "log_sizes",
        "yaws"
    ]:
        module = getattr(head, name)
        grad = module.weight.grad

        valid = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}分支有效梯度:", valid)
        assert valid

    assert bev.grad is not None
    assert torch.isfinite(bev.grad).all()
    assert bev.grad.abs().sum().item() > 0

    print("\nBEV检测头与损失函数测试通过")


if __name__ == "__main__":
    main()
