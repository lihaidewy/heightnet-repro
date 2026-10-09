
import math
import torch

from src.heightnet.center_head_full import CenterHeadFull
from src.heightnet.center_targets_full import CenterTargetGenerator
from src.heightnet.center_losses_full import CenterPointLoss


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")

    # 标准化的合成世界坐标车辆框
    # [x,y,z_bottom,length,width,height,yaw]
    boxes = [
        torch.tensor([
            [44.25, -28.75, -30.0, 5.0, 2.2, 1.8, 0.4],
            [167.2, 47.2, -30.0, 4.8, 2.1, 1.7, 0.0],
        ]),
        torch.tensor([
            [80.5, 0.25, -30.0, 5.1, 2.3, 1.9, -0.7],
            # 超出X上界，应被过滤
            [168.0, 0.0, -30.0, 5.0, 2.2, 1.8, 0.0],
        ]),
    ]

    generator = CenterTargetGenerator()
    targets = generator(boxes, device)

    print(
        "Heatmap:",
        tuple(targets["heatmap"].shape)
    )
    print(
        "Annotations:",
        tuple(targets["anno"].shape)
    )

    num_gt = targets["mask"].sum().item()
    print("有效GT数量:", num_gt)

    assert num_gt == 3
    assert targets["heatmap"].shape == (
        2, 1, 80, 128
    )

    # 第一辆车应落在 y=3, x=4
    assert targets["heatmap"][0, 0, 3, 4] == 1

    expected = torch.tensor(
        [
            0.25,
            0.25,
            0.0,
            math.log(5.0),
            math.log(2.2),
            math.log(1.8),
            math.sin(0.4),
            math.cos(0.4),
        ],
        device=device,
    )

    actual = targets["anno"][0, 0]

    assert torch.allclose(
        actual, expected, atol=1e-5
    )

    print("GT坐标与回归编码检查通过")

    # 使用真实的 CenterHeadFull 检验损失
    features = torch.randn(
        2, 256, 80, 128,
        device=device,
        requires_grad=True,
    )

    head = CenterHeadFull(
        with_velocity=False
    ).to(device)

    predictions = head(features)

    criterion = CenterPointLoss().to(device)
    losses = criterion(
        predictions, targets
    )

    for name, value in losses.items():
        print(f"{name} loss: {value.item():.6f}")
        assert torch.isfinite(value)

    assert losses["total"].item() > 0

    losses["total"].backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0

    print("BEV特征梯度正常: True")

    for name in [
        "heatmap", "reg", "height", "dim", "rot"
    ]:
        grad = (
            head.task_head.branches[name][-1]
            .weight.grad
        )

        ok = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}梯度正常: {ok}")
        assert ok

    # 空标签场景不应产生NaN
    empty = generator(
        [torch.empty(0, 7, device=device)],
        device,
    )

    empty_features = torch.randn(
        1, 256, 80, 128, device=device
    )

    head.eval()
    with torch.no_grad():
        empty_pred = head(empty_features)
        empty_loss = criterion(
            empty_pred, empty
        )["total"]

    assert torch.isfinite(empty_loss)
    print("空标签损失检查通过")

    print(
        "GPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2,
            2,
        ),
    )

    print(
        "\nCenterPoint目标生成与损失测试全部通过"
    )


if __name__ == "__main__":
    main()
