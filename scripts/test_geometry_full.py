
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.geometry_full import (
    HeightGeometryFull,
)


def main():
    device = "cuda"

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

    world2cam = torch.as_tensor(
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

    result = geometry(
        K=K,
        world2cam=world2cam,
        image_hw=(576, 1024),
        feature_hw=(36, 64),
        ground_z=-30.0,
    )

    xyz = result["xyz"]
    valid = result["valid"]
    heights = result["heights"]

    print("XYZ:", tuple(xyz.shape))
    print("Valid:", tuple(valid.shape))
    print(
        "高度范围:",
        heights.min().item(),
        heights.max().item(),
    )

    valid_count = valid.sum().item()
    total_count = valid.numel()

    print(
        "有效几何点:",
        valid_count, "/", total_count,
    )

    assert xyz.shape == (1, 90, 36, 64, 3)
    assert valid.shape == (1, 90, 36, 64)
    assert valid_count > 0

    # 检查三维点是否位于对应的高度平面
    expected_z = (
        -30.0
        + heights.view(1, 90, 1, 1)
    )

    z_error = (
        xyz[..., 2] - expected_z
    ).abs()

    max_z_error = z_error[valid].max().item()

    print("最大高度平面误差(m):", max_z_error)
    assert max_z_error < 1e-3

    # 三维世界点重新投影到原始相机
    R = world2cam[:, :3, :3]
    t = world2cam[:, :3, 3]

    cam_points = (
        torch.einsum(
            "bij,bdhwj->bdhwi",
            R, xyz,
        )
        + t[:, None, None, None, :]
    )

    X = cam_points[..., 0]
    Y = cam_points[..., 1]
    Z = cam_points[..., 2]

    safe_X = torch.where(
        valid, X, torch.ones_like(X)
    )

    u_reproject = (
        K[:, 0, 2, None, None, None]
        - K[:, 0, 0, None, None, None]
        * Y / safe_X
    )

    v_reproject = (
        K[:, 1, 2, None, None, None]
        - K[:, 1, 1, None, None, None]
        * Z / safe_X
    )

    u_expected = result["pixel_u"].unsqueeze(1)
    v_expected = result["pixel_v"].unsqueeze(1)

    reproj_error = torch.maximum(
        (u_reproject - u_expected).abs(),
        (v_reproject - v_expected).abs(),
    )

    max_reproj = reproj_error[valid].max().item()

    print("最大重投影误差(pixel):", max_reproj)
    assert max_reproj < 0.05

    # 独立检查 Y 左坐标方向
    identity_T = torch.eye(
        4, device=device
    ).unsqueeze(0)

    synthetic = geometry(
        K=K,
        world2cam=identity_T,
        image_hw=(576, 1024),
        feature_hw=(36, 64),
        ground_z=-30.0,
    )

    row = 30
    col = 10

    assert synthetic["valid"][0, 0, row, col]

    # 像素位于图像中心左侧：
    # 根据已确认的相机约定，应对应 Y > 0
    u = synthetic["pixel_u"][0, row, col]

    assert u < K[0, 0, 2]
    assert synthetic["xyz"][0, 0, row, col, 1] > 0

    print("相机Y左方向检查通过")

    print("完整高度几何测试通过")


if __name__ == "__main__":
    main()
