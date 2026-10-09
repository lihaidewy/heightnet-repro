import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.height_bins import LIDHeightBins
from src.heightnet.geometry import (
    build_height_frustum,
    project_world_to_pixel,
)


def main():
    dataset = HeightNetDataset("data/S3S")

    # 选择之前验证过的 257 帧
    index = next(
        i for i, (frame_id, _, _) in enumerate(dataset.samples)
        if frame_id == "257"
    )

    sample = dataset[index]

    K = sample["intrinsics"].unsqueeze(0)
    T = sample["extrinsics_raw"].unsqueeze(0)

    lid = LIDHeightBins(
        num_bins=90,
        h_min=0.0,
        h_max=4.0
    )

    xyz, valid = build_height_frustum(
        K=K,
        T=T,
        height_centers=lid.centers,
        feature_hw=(36, 64),
        image_hw=(576, 1024),
        ground_z=-30.0
    )

    print("三维坐标网格:", xyz.shape)
    print("有效位置掩码:", valid.shape)
    print("有效候选比例:", valid.float().mean().item())

    # 利用真实框的底面中心检查投影位置
    gt_point = sample["boxes_raw"][:1, :3]
    pixels, point_valid = project_world_to_pixel(
        gt_point,
        K[0],
        T[0]
    )

    u, v = pixels[0].tolist()

    row = round((v + 0.5) * 36 / 576 - 0.5)
    col = round((u + 0.5) * 64 / 1024 - 0.5)

    assert point_valid[0]
    assert 0 <= row < 36
    assert 0 <= col < 64

    # 第 0 个分箱接近地面高度 0
    reconstructed = xyz[0, 0, row, col]
    error = torch.linalg.norm(reconstructed - gt_point[0])

    print("真实目标底面中心:", gt_point[0])
    print("最近特征位置:", (row, col))
    print("对应三维位置:", reconstructed)
    print("近似位置误差:", error.item(), "m")

    assert xyz.shape == (1, 90, 36, 64, 3)
    assert valid.shape == (1, 90, 36, 64)
    assert torch.isfinite(xyz).all()

    print("高度视锥三维投影测试通过")


if __name__ == "__main__":
    main()