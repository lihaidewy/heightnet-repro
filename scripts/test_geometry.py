
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.geometry import (
    project_world_to_pixel,
    pixel_height_to_world
)


def main():
    dataset = HeightNetDataset("data/S3S")

    # 查找 257 帧
    index = next(
        i for i, (frame_id, _, _) in enumerate(dataset.samples)
        if frame_id == "257"
    )

    sample = dataset[index]

    K = sample["intrinsics"]
    T = sample["extrinsics_raw"]

    # 第一个车辆 3D Box 底面中心
    point_world = sample["boxes_raw"][:1, :3]

    print("原始世界坐标:", point_world)

    # 世界坐标 -> 图像像素
    pixels, valid1 = project_world_to_pixel(
        point_world, K, T
    )

    print("投影像素:", pixels)

    # 假设参考地面为 -30 m
    # 底面中心的离地高度为 0 m
    heights = torch.zeros(1)

    # 图像像素 + 高度 -> 世界坐标
    reconstructed, valid2 = pixel_height_to_world(
        pixels, heights, K, T, ground_z=-30.0
    )

    print("恢复世界坐标:", reconstructed)

    error = torch.linalg.norm(
        reconstructed - point_world,
        dim=-1
    )

    print("三维恢复误差:", error.item(), "m")

    assert valid1.all()
    assert valid2.all()
    assert error.item() < 1e-3

    print("Height-to-3D 几何验证通过")


if __name__ == "__main__":
    main()
