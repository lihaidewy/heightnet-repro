
import json
from pathlib import Path
from collections import Counter

from src.heightnet.dataset import HeightNetDataset


# 使用固定时间区间，保证实验可复现
VAL_INTERVALS = [
    (290, 329),
    (450, 489),
]

# 验证区间前后隔离的帧数
GAP = 10

OUTPUT_DIR = Path("data/splits")


def assign_split(frame_id):
    """根据全局帧号决定数据用途。"""

    fid = int(frame_id)

    # 验证集优先
    for start, end in VAL_INTERVALS:
        if start <= fid <= end:
            return "val"

    # 与验证区间过近的帧不参与训练
    for start, end in VAL_INTERVALS:
        if start - GAP <= fid <= end + GAP:
            return "excluded"

    return "train"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": [],
        "val": [],
        "excluded": [],
    }

    # 依次处理两个数据集
    for scene in ["S3S", "S3M"]:
        dataset = HeightNetDataset(
            f"data/{scene}",
            image_size=(576, 1024),
        )

        counter = Counter()

        for frame_id, _, _ in dataset.samples:
            split = assign_split(frame_id)

            sample_info = {
                "scene": scene,
                "frame_id": str(frame_id),
            }

            splits[split].append(sample_info)
            counter[split] += 1

        print(f"\n========== {scene} ==========")
        print("训练集:", counter["train"])
        print("验证集:", counter["val"])
        print("隔离帧:", counter["excluded"])

    # 检查相同帧号不会横跨训练和验证集
    train_ids = {
        int(x["frame_id"])
        for x in splits["train"]
    }

    val_ids = {
        int(x["frame_id"])
        for x in splits["val"]
    }

    assert train_ids.isdisjoint(val_ids)

    # 检查训练与验证帧号间隔
    for train_id in train_ids:
        for val_id in val_ids:
            assert abs(train_id - val_id) > GAP

    # 检查无重复样本
    all_keys = [
        (x["scene"], x["frame_id"])
        for records in splits.values()
        for x in records
    ]

    assert len(all_keys) == len(set(all_keys))

    # 保存三个数据清单
    for name, records in splits.items():
        path = OUTPUT_DIR / f"{name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                records,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"\n{name}: {len(records)} 帧")
        print("保存位置:", path)

    # 同时保存划分规则
    config = {
        "val_intervals": VAL_INTERVALS,
        "gap_frames": GAP,
        "scenes": ["S3S", "S3M"],
    }

    with open(
        OUTPUT_DIR / "split_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    print("\n训练集/验证集划分检查通过")


if __name__ == "__main__":
    main()
