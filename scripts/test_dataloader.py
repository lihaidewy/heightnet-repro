
from collections import Counter

import torch
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import (
    SplitHeightNetDataset,
    heightnet_collate_fn,
)
from src.heightnet.targets import BEVTargetEncoder


def main():
    torch.manual_seed(42)

    # 1. 读取划分后的两个数据集
    train_dataset = SplitHeightNetDataset(
        "data/splits/train.json",
        image_size=(576, 1024),
    )

    val_dataset = SplitHeightNetDataset(
        "data/splits/val.json",
        image_size=(576, 1024),
    )

    assert len(train_dataset) == 395
    assert len(val_dataset) == 114

    # 2. 确认各场景样本数量
    for name, dataset in [
        ("训练集", train_dataset),
        ("验证集", val_dataset),
    ]:
        counter = Counter(
            scene for scene, _, _ in dataset.records
        )
        print(f"{name}场景分布:", dict(counter))

    # 3. 确认相同帧号不会跨训练/验证集合
    train_ids = {
        fid for _, fid, _ in train_dataset.records
    }

    val_ids = {
        fid for _, fid, _ in val_dataset.records
    }

    assert train_ids.isdisjoint(val_ids)

    # 4. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=heightnet_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=heightnet_collate_fn,
        drop_last=False,
    )

    print("训练集Batch数量:", len(train_loader))
    print("验证集Batch数量:", len(val_loader))

    # 5. 读取第一个 Batch
    batch = next(iter(train_loader))

    print("\nBatch检查:")
    print("场景:", batch["scene"])
    print("帧号:", batch["frame_id"])
    print("RGB:", batch["image"].shape)
    print("内参:", batch["intrinsics"].shape)
    print("外参:", batch["extrinsics_raw"].shape)

    print(
        "每帧车辆数量:",
        [len(boxes) for boxes in batch["boxes_raw"]]
    )

    assert batch["image"].shape == (
        2, 3, 576, 1024
    )

    assert batch["intrinsics"].shape == (2, 3, 3)
    assert batch["extrinsics_raw"].shape == (2, 4, 4)

    # 6. 检查批量标注能否转换为检测监督
    encoder = BEVTargetEncoder(
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0,
    )

    targets = [
        encoder(boxes)
        for boxes in batch["boxes_raw"]
    ]

    print("\n检测监督检查:")
    for i, target in enumerate(targets):
        print(
            f"样本{i}:",
            f"Heatmap={tuple(target['heatmap'].shape)}",
            f"车辆={len(target['indices'])}"
        )

        assert target["heatmap"].shape == (
            1, 80, 128
        )

        assert len(target["indices"]) == len(
            batch["boxes_raw"][i]
        )

    print("\n多场景DataLoader测试通过")


if __name__ == "__main__":
    main()
