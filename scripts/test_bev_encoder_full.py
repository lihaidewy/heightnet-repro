
import torch

from src.heightnet.bev_encoder_full import (
    BEVEncoderFull,
)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda")

    # 模拟前一阶段真实网络输出的BEV尺寸
    bev = torch.randn(
        2, 80, 80, 128,
        device=device,
        requires_grad=True,
    )

    model = BEVEncoderFull().to(device)
    model.train()

    stage_shapes = {}
    handles = []

    for name in ["layer1", "layer2", "layer3"]:
        layer = getattr(model, name)

        def make_hook(stage_name):
            def hook(module, inputs, output):
                stage_shapes[stage_name] = tuple(
                    output.shape
                )
            return hook

        handles.append(
            layer.register_forward_hook(
                make_hook(name)
            )
        )

    torch.cuda.reset_peak_memory_stats(device)

    output = model(bev)

    for handle in handles:
        handle.remove()

    print("输入BEV:", tuple(bev.shape))

    for name, shape in stage_shapes.items():
        print(name, shape)

    print("融合BEV:", tuple(output.shape))

    assert stage_shapes["layer1"] == (
        2, 160, 40, 64
    )

    assert stage_shapes["layer2"] == (
        2, 320, 20, 32
    )

    assert stage_shapes["layer3"] == (
        2, 640, 10, 16
    )

    assert output.shape == (
        2, 256, 80, 128
    )

    assert torch.isfinite(output).all()

    # 检查完整反向传播
    loss = output.square().mean()
    loss.backward()

    checks = {
        "BEV输入": bev,
        "Stem": model.stem[0].weight,
        "Stage1": model.layer1[0].conv1.weight,
        "Stage2": model.layer2[0].conv1.weight,
        "Stage3": model.layer3[0].conv1.weight,
        "FPN0": model.fpn.deblocks[0][0].weight,
        "FPN1": model.fpn.deblocks[1][0].weight,
        "FPN2": model.fpn.deblocks[2][0].weight,
        "FPN3": model.fpn.deblocks[3][0].weight,
    }

    print("\n梯度检查:")

    for name, tensor in checks.items():
        grad = tensor.grad

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
        "\nBEV Encoder参数量:",
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
        "\n完整BEV ResNet18 + SECOND-FPN测试通过"
    )


if __name__ == "__main__":
    main()
