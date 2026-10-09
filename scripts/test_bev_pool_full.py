
import torch

from src.heightnet.bev_pool_full import VoxelPoolingFull
from src.heightnet.geometry_full import HeightGeometryFull
from src.heightnet.dataset import HeightNetDataset


def test_synthetic(device):
    """
    使用手动设置的三维点，
    独立计算参考结果。
    """
    pool = VoxelPoolingFull(
        x_bound=(40.0, 43.0, 1.0),
        y_bound=(-32.0, -29.0, 1.0),
        z_bound=(-30.0, -26.0, 4.0),
        chunk_bins=1,
    ).to(device)

    context = torch.tensor(
        [[
            [[1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0]],
        ]],
        device=device,
        requires_grad=True,
    )

    logits = torch.tensor(
        [[
            [[0.2, -0.4, 0.0]],
            [[1.1, 0.6, -0.1]],
        ]],
        device=device,
        requires_grad=True,
    )

    prob = torch.softmax(logits, dim=1)

    # [B=1,D=2,H=1,W=3,XYZ=3]
    xyz = torch.zeros(
        (1, 2, 1, 3, 3), device=device
    )

    # 两个不同高度的点落入同一 BEV 网格
    xyz[0, 0, 0, 0] = torch.tensor(
        [40.2, -31.8, -29.8], device=device
    )
    xyz[0, 1, 0, 0] = torch.tensor(
        [40.9, -31.1, -28.8], device=device
    )

    # 相同 X、不同 Y 网格
    xyz[0, 0, 0, 1] = torch.tensor(
        [41.3, -31.3, -29.5], device=device
    )
    xyz[0, 1, 0, 1] = torch.tensor(
        [41.3, -30.6, -28.2], device=device
    )

    # 恰好位于 X 上界，应排除
    xyz[0, 0, 0, 2] = torch.tensor(
        [43.0, -31.0, -29.0], device=device
    )

    # 超出 Z 上界，应排除
    xyz[0, 1, 0, 2] = torch.tensor(
        [42.2, -29.9, -25.9], device=device
    )

    valid = torch.ones(
        (1, 2, 1, 3),
        dtype=torch.bool,
        device=device,
    )

    bev = pool(context, prob, xyz, valid)

    print("合成数据BEV:", tuple(bev.shape))
    assert bev.shape == (1, 2, 3, 3)

    # 使用简单循环独立计算参考结果
    reference = torch.zeros_like(bev)

    for d in range(2):
        for w in range(3):
            x, y, z = xyz[0, d, 0, w].tolist()

            if not (
                40.0 <= x < 43.0
                and -32.0 <= y < -29.0
                and -30.0 <= z < -26.0
            ):
                continue

            ix = int((x - 40.0) // 1.0)
            iy = int((y + 32.0) // 1.0)

            reference[0, :, iy, ix] += (
                context.detach()[0, :, 0, w]
                * prob.detach()[0, d, 0, w]
            )

    assert torch.allclose(
        bev.detach(),
        reference,
        atol=1e-6,
        rtol=1e-6,
    )

    print("手工参考结果对比通过")

    # 检查反向传播
    loss = bev.square().mean()
    loss.backward()

    assert context.grad is not None
    assert logits.grad is not None

    assert torch.isfinite(context.grad).all()
    assert torch.isfinite(logits.grad).all()

    assert context.grad.abs().sum() > 0
    assert logits.grad.abs().sum() > 0

    print("Context梯度有效: True")
    print("Height Logits梯度有效: True")


def test_real_geometry(device):
    """
    使用 S3S 真实相机参数验证
    Geometry -> Voxel Pooling。
    此处暂用模拟图像特征。
    """
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    sample = dataset[0]

    K = torch.as_tensor(
        sample["intrinsics"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    T = torch.as_tensor(
        sample["extrinsics_raw"],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    geometry = HeightGeometryFull(
        num_bins=90,
        height_min=0.0,
        height_max=4.0,
        height_method="lid",
        pixel_mode="official",
    ).to(device)

    pool = VoxelPoolingFull(
        chunk_bins=12
    ).to(device)

    torch.manual_seed(42)

    context = torch.randn(
        1, 80, 36, 64, device=device
    )

    logits = torch.randn(
        1, 90, 36, 64, device=device
    )

    prob = torch.softmax(logits, dim=1)

    with torch.no_grad():
        geom = geometry(
            K=K,
            world2cam=T,
            image_hw=(576, 1024),
            feature_hw=(36, 64),
            ground_z=-30.0,
        )

        bev = pool(
            context,
            prob,
            geom["xyz"],
            geom["valid"],
        )

    print("真实几何BEV:", tuple(bev.shape))

    assert bev.shape == (1, 80, 80, 128)
    assert torch.isfinite(bev).all()

    occupancy = (
        bev.abs().sum(dim=1) > 1e-6
    )

    nonzero = occupancy.sum().item()

    print("非空BEV网格数:", nonzero)

    assert nonzero > 0

    print("真实几何BEV Pooling测试通过")


def main():
    device = torch.device("cuda")

    test_synthetic(device)
    test_real_geometry(device)

    print(
        "GPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2, 2
        ),
    )

    print("完整Voxel Pooling测试通过")


if __name__ == "__main__":
    main()
