
import json
from pathlib import Path

import torch

from src.heightnet.targets import BEVTargetEncoder


def main():
    encoder = BEVTargetEncoder()

    for scene in ["S3S", "S3M"]:
        label_dir = Path("data") / scene / "labels"

        total_boxes = 0
        encoded_boxes = 0
        collisions = 0
        max_error = 0.0

        for path in sorted(label_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                ann = json.load(f)

            rows = []

            for obj in ann.get("objects", []):
                if obj.get("category") != "car":
                    continue

                rows.append(
                    list(obj["translation"])
                    + list(obj["size"])
                    + [obj["rotation"]]
                )

            boxes = torch.tensor(
                rows, dtype=torch.float32
            ).reshape(-1, 7)

            targets = encoder(boxes)

            n = len(targets["indices"])
            total_boxes += len(boxes)
            encoded_boxes += n

            # 检查中心回归是否有网格冲突
            unique_count = len(
                torch.unique(targets["indices"])
            )
            collisions += n - unique_count

            # 检查中心坐标编码后能否精确恢复
            if n == len(boxes) and n > 0:
                ix = targets["indices"] % encoder.nx
                iy = targets["indices"] // encoder.nx

                recovered_x = (
                    ix.float() + targets["offsets"][:, 0]
                ) * encoder.resolution + encoder.x_min

                recovered_y = (
                    iy.float() + targets["offsets"][:, 1]
                ) * encoder.resolution + encoder.y_min

                recovered = torch.stack(
                    [recovered_x, recovered_y], dim=1
                )

                error = torch.linalg.norm(
                    recovered - boxes[:, :2], dim=1
                ).max().item()

                max_error = max(max_error, error)

        print(f"\n========== {scene} ==========")
        print("车辆总数:", total_boxes)
        print("成功编码:", encoded_boxes)
        print("同一网格中心冲突:", collisions)
        print("中心恢复最大误差(m):", max_error)
        print(
            "Heatmap尺寸:",
            (1, encoder.ny, encoder.nx)
        )

        assert encoded_boxes == total_boxes
        assert max_error < 1e-3

    print("\nBEV检测标签编码测试通过")


if __name__ == "__main__":
    main()
