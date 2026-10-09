
import torch

from src.heightnet.center_head_full import CenterHeadFull


def test_head(device, with_velocity):
    torch.manual_seed(42)

    features = torch.randn(
        2, 256, 80, 128,
        device=device,
        requires_grad=True,
    )

    model = CenterHeadFull(
        in_channels=256,
        shared_channels=64,
        num_classes=1,
        with_velocity=with_velocity,
    ).to(device)

    model.train()

    outputs = model(features)

    expected_channels = {
        "heatmap": 1,
        "reg": 2,
        "height": 1,
        "dim": 3,
        "rot": 2,
    }

    if with_velocity:
        expected_channels["vel"] = 2

    print(
        "\nVelocity分支:",
        "开启" if with_velocity else "关闭",
    )

    for name, channels in expected_channels.items():
        output = outputs[name]

        print(name, tuple(output.shape))

        assert output.shape == (
            2, channels, 80, 128
        )
        assert torch.isfinite(output).all()

    assert set(outputs.keys()) == set(
        expected_channels.keys()
    )

    # 检查官方风格的热力图偏置初始化
    bias = (
        model.task_head
        .branches["heatmap"][-1].bias
    )

    assert torch.allclose(
        bias,
        torch.full_like(bias, -2.19),
    )

    print("Heatmap初始化检查通过")

    # 独立测试损失，仅用于检查梯度
    loss = sum(
        output.square().mean()
        for output in outputs.values()
    )

    loss.backward()

    # BEV输入必须有梯度
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0

    # 共享卷积必须有梯度
    shared_grad = (
        model.shared_conv[0].weight.grad
    )

    assert shared_grad is not None
    assert torch.isfinite(shared_grad).all()
    assert shared_grad.abs().sum() > 0

    print("BEV输入梯度有效: True")
    print("共享卷积梯度有效: True")

    # 每一个独立分支都必须收到梯度
    for name in expected_channels:
        parameter = (
            model.task_head.branches[name][-1].weight
        )

        grad = parameter.grad

        ok = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )

        print(f"{name}分支梯度有效: {ok}")
        assert ok

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print("检测头参数量:", f"{parameters:,}")

    print("本轮检测头测试通过")

    del model, features, outputs, loss


def main():
    device = torch.device("cuda")

    torch.cuda.reset_peak_memory_stats(device)

    # S3 当前实际可训练配置
    test_head(device, with_velocity=False)

    # 检查完整结构能够支持速度预测
    test_head(device, with_velocity=True)

    print(
        "\nGPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated(device)
            / 1024**2,
            2,
        ),
    )

    print("\nCenterPoint风格检测头测试全部通过")


if __name__ == "__main__":
    main()
