
import json
from pathlib import Path
from collections import Counter

import numpy as np


X_MIN, X_MAX = 0.0, 160.0
Y_MIN, Y_MAX = -80.0, 80.0


def inspect_scene(scene):
    label_dir = Path("data") / scene / "labels"

    print(f"\n========== {scene} ==========")

    if not label_dir.is_dir():
        print("标注目录不存在:", label_dir)
        return

    files = sorted(label_dir.glob("*.json"))
    print("JSON文件数量:", len(files))

    records = []
    categories = Counter()
    invalid_count = 0

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        for obj in ann.get("objects", []):
            try:
                xyz = np.asarray(
                    obj["translation"], dtype=float
                )
                size = np.asarray(
                    obj["size"], dtype=float
                )

                if (
                    xyz.shape != (3,)
                    or size.shape != (3,)
                    or not np.isfinite(xyz).all()
                    or not np.isfinite(size).all()
                ):
                    invalid_count += 1
                    continue

                records.append((path.stem, xyz, size))
                categories[obj.get("category", "unknown")] += 1

            except (KeyError, TypeError, ValueError):
                invalid_count += 1

    print("有效3D Box数量:", len(records))
    print("无效3D Box数量:", invalid_count)
    print("类别分布:", dict(categories))

    if not records:
        return

    positions = np.stack([r[1] for r in records])
    sizes = np.stack([r[2] for r in records])

    for axis, name in enumerate(["X", "Y", "Z"]):
        values = positions[:, axis]
        print(
            f"{name}坐标:",
            f"min={values.min():.3f},",
            f"median={np.median(values):.3f},",
            f"max={values.max():.3f}"
        )

    print("3D Box尺寸中位数(L,W,H):",
          np.median(sizes, axis=0).round(3))

    # 检查当前 BEV 范围
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    inside = (
        (x >= X_MIN) & (x < X_MAX)
        & (y >= Y_MIN) & (y < Y_MAX)
    )

    print(
        "当前BEV范围内的Box中心:",
        f"{inside.sum()}/{len(inside)}",
        f"({inside.mean() * 100:.2f}%)"
    )

    # 检查当前使用的地面高度
    near_ground = np.abs(z + 30.0) < 0.1

    print(
        "底面Z接近-30m的比例:",
        f"{near_ground.mean() * 100:.2f}%"
    )

    print(
        "Z坐标分位数(5%,50%,95%):",
        np.percentile(z, [5, 50, 95]).round(3)
    )

    # 展示少量不在当前BEV范围内的目标
    outside_idx = np.where(~inside)[0][:5]

    for idx in outside_idx:
        frame_id, xyz, _ = records[idx]
        print(
            "BEV范围外样本:",
            frame_id,
            np.round(xyz, 3).tolist()
        )


if __name__ == "__main__":
    inspect_scene("S3S")
    inspect_scene("S3M")
