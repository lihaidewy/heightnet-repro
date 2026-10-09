
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HeightNetDataset(Dataset):
    """RGB 图像与 3D Box JSON 配对数据集。"""

    def __init__(self, scene_dir, image_size=(576, 1024)):
        self.scene_dir = Path(scene_dir)
        self.image_dir = self.scene_dir / "images"
        self.label_dir = self.scene_dir / "labels"

        # image_size = (height, width)
        self.image_size = image_size

        if not self.image_dir.is_dir():
            raise FileNotFoundError(self.image_dir)
        if not self.label_dir.is_dir():
            raise FileNotFoundError(self.label_dir)

        image_files = {
            p.stem: p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }

        label_files = {
            p.stem: p
            for p in self.label_dir.glob("*.json")
        }

        frame_ids = set(image_files) & set(label_files)

        self.samples = [
            (frame_id, image_files[frame_id], label_files[frame_id])
            for frame_id in sorted(
                frame_ids,
                key=lambda x: int(x) if x.isdigit() else x
            )
        ]

        print(
            f"[{self.scene_dir.name}] "
            f"Matched samples: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_id, image_path, label_path = self.samples[index]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        orig_w, orig_h = image.size
        target_h, target_w = self.image_size

        with open(label_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        # 1. 相机内参
        intr = ann["intrinsics"]

        K = np.array([
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 2. 缩放图像和内参
        sx = target_w / orig_w
        sy = target_h / orig_h

        K[0, :] *= sx
        K[1, :] *= sy

        image = image.resize(
            (target_w, target_h),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(
            image, dtype=np.float32
        ) / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1).copy()
        )

        # 3. 保留原始外参
        # 暂时不假设矩阵的变换方向
        T_raw = np.asarray(
            ann["extrinsics_matrix_4x4"],
            dtype=np.float32
        )

        # 4. 读取原始 3D Bounding Boxes
        boxes = []
        categories = []

        for obj in ann.get("objects", []):
            box = [
                *obj["translation"],
                *obj["size"],
                obj["rotation"]
            ]
            boxes.append(box)
            categories.append(obj["category"])

        boxes = np.asarray(
            boxes, dtype=np.float32
        ).reshape(-1, 7)

        return {
            "frame_id": frame_id,
            "image": image,
            "intrinsics": torch.from_numpy(K),
            "extrinsics_raw": torch.from_numpy(T_raw),
            "boxes_raw": torch.from_numpy(boxes),
            "categories": categories,
            "original_size": (orig_h, orig_w),
        }


def heightnet_collate_fn(batch):
    """允许每帧具有不同数量的目标。"""
    return {
        "frame_id": [x["frame_id"] for x in batch],
        "image": torch.stack([x["image"] for x in batch]),
        "intrinsics": torch.stack(
            [x["intrinsics"] for x in batch]
        ),
        "extrinsics_raw": torch.stack(
            [x["extrinsics_raw"] for x in batch]
        ),
        "boxes_raw": [x["boxes_raw"] for x in batch],
        "categories": [x["categories"] for x in batch],
        "original_size": [x["original_size"] for x in batch],
    }
