
import argparse

import torch
import torch.nn.functional as F

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.height_bins import LIDHeightBins
from src.heightnet.geometry import build_height_frustum


# 与检测标签编码器保持一致
X_MIN, X_MAX = 40.0, 168.0
Y_MIN, Y_MAX = -32.0, 48.0
RESOLUTION = 1.0

NX = int((X_MAX - X_MIN) / RESOLUTION)
NY = int((Y_MAX - Y_MIN) / RESOLUTION)


@torch.no_grad()
def check_frame(sample, height_centers):
    K = sample["intrinsics"].unsqueeze(0)
    T = sample["extrinsics_raw"].unsqueeze(0)

    # 当前使用 288x512 图像，对应 18x32 特征图
    xyz, valid = build_height_frustum(
        K=K,
        T=T,
        height_centers=height_centers,
        feature_hw=(36, 64),
        image_hw=(576, 1024),
        ground_z=-30.0,
    )

    x = xyz[0, ..., 0]
    y = xyz[0, ..., 1]

    ix = torch.floor(
        (x - X_MIN) / RESOLUTION
    ).long()

    iy = torch.floor(
        (y - Y_MIN) / RESOLUTION
    ).long()

    mask = (
        valid[0]
        & torch.isfinite(x)
        & torch.isfinite(y)
        & (ix >= 0)
        & (ix < NX)
        & (iy >= 0)
        & (iy < NY)
    )

    # 只统计几何投影位置，不依赖随机网络特征
    indices = (iy * NX + ix)[mask]

    counts = torch.bincount(
        indices, minlength=NY * NX
    ).reshape(NY, NX)

    occupied = counts > 0

    # 允许检测头利用邻近网格信息
    def dilate(mask_2d, kernel_size):
        result = F.max_pool2d(
            mask_2d.float()[None, None],
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        return result[0, 0] > 0

    near_3x3 = dilate(occupied, 3)
    near_5x5 = dilate(occupied, 5)

    boxes = sample["boxes_raw"]

    gx = torch.floor(
        (boxes[:, 0] - X_MIN) / RESOLUTION
    ).long()

    gy = torch.floor(
        (boxes[:, 1] - Y_MIN) / RESOLUTION
    ).long()

    inside = (
        (gx >= 0) & (gx < NX)
        & (gy >= 0) & (gy < NY)
    )

    gx = gx[inside]
    gy = gy[inside]

    n = len(gx)

    return {
        "total": len(boxes),
        "inside": n,
        "exact": int(occupied[gy, gx].sum()),
        "near3": int(near_3x3[gy, gx].sum()),
        "near5": int(near_5x5[gy, gx].sum()),
        "occupied_cells": int(occupied.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true",
        help="检查所有标注帧；默认抽样检查"
    )
    args = parser.parse_args()

    lid = LIDHeightBins(
        num_bins=90,
        h_min=0.0,
        h_max=4.0,
    )

    for scene in ["S3S", "S3M"]:
        dataset = HeightNetDataset(
            f"data/{scene}",
            image_size=(576, 1024),
        )

        if args.all:
            selected = range(len(dataset))
        else:
            selected = {
                0,
                len(dataset) // 4,
                len(dataset) // 2,
                3 * len(dataset) // 4,
                len(dataset) - 1,
            }

            # 额外检查已经完成几何验证的257帧
            for i, (fid, _, _) in enumerate(dataset.samples):
                if str(fid) == "257":
                    selected.add(i)
                    break

            selected = sorted(selected)

        totals = {
            "total": 0,
            "inside": 0,
            "exact": 0,
            "near3": 0,
            "near5": 0,
        }

        print(f"\n========== {scene} ==========")

        for i in selected:
            sample = dataset[i]
            result = check_frame(
                sample, lid.centers
            )

            for key in totals:
                totals[key] += result[key]

            print(
                f"帧 {sample['frame_id']}: "
                f"车辆 {result['inside']}, "
                f"精确覆盖 {result['exact']}, "
                f"3x3覆盖 {result['near3']}, "
                f"5x5覆盖 {result['near5']}, "
                f"非空BEV网格 {result['occupied_cells']}"
            )

        n = totals["inside"]

        print("\n汇总:")
        print("车辆总数:", totals["total"])
        print("BEV范围内车辆:", n)

        for key, name in [
            ("exact", "精确覆盖率"),
            ("near3", "3x3邻域覆盖率"),
            ("near5", "5x5邻域覆盖率"),
        ]:
            percentage = (
                100.0 * totals[key] / n
                if n > 0 else 0.0
            )
            print(
                f"{name}: "
                f"{totals[key]}/{n} "
                f"({percentage:.2f}%)"
            )


if __name__ == "__main__":
    main()
