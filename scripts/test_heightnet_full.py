
import torch

from src.heightnet.heightnet_full import (
    HeightNetFull,
    build_camera_condition,
)


def main():
    device = "cuda"
    B, H, W = 2, 9, 16

    features = torch.randn(
        B, 512, H, W,
        device=device,
        requires_grad=True,
    )

    K = torch.eye(3, device=device).repeat(B, 1, 1)
    K[:, 0, 0] = 886.8101
    K[:, 1, 1] = 886.8101
    K[:, 0, 2] = 512.0
    K[:, 1, 2] = 288.0
    K[1, 0, 0] = 890.0

    world2cam = torch.eye(
        4, device=device
    ).repeat(B, 1, 1)
    world2cam[1, 0, 3] = 0.2

    condition = build_camera_condition(
        K, world2cam
    )

    print("相机条件:", tuple(condition.shape))
    print("条件是否有限:", torch.isfinite(condition).all().item())

    assert condition.shape == (B, 27)
    assert torch.isfinite(condition).all()

    model = HeightNetFull().to(device)
    model.train()

    output = model(
        features, K, world2cam
    )

    print("HeightNet输出:", tuple(output.shape))
    assert output.shape == (B, 170, H, W)

    logits = output[:, :90]
    context = output[:, 90:]

    prob = torch.softmax(logits, dim=1)

    print("高度logits:", tuple(logits.shape))
    print("Context:", tuple(context.shape))

    assert torch.allclose(
        prob.sum(dim=1),
        torch.ones_like(prob[:, 0]),
        atol=1e-5,
    )

    loss = output.square().mean()
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()

    print("输入特征梯度正常: True")

    param_count = sum(
        p.numel() for p in model.parameters()
    )
    print("HeightNet参数量:", f"{param_count:,}")

    print(
        "GPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        ),
    )

    print("完整HeightNet前向反向测试通过")


if __name__ == "__main__":
    main()
