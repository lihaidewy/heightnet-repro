
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import (
    SplitHeightNetDataset,
    heightnet_collate_fn,
)
from src.heightnet.roadside_detector import RoadsideVehicleDetector
import math


def percentile90(errors):
    """Nearest-rank 90th percentile."""
    if not errors:
        return float("nan")

    values = sorted(errors)
    index = math.ceil(0.90 * len(values)) - 1
    return values[index]

@torch.inference_mode()
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        "outputs/train_full/best.pth",
        map_location="cpu",
        weights_only=True,
    )

    print("加载模型 Epoch:", checkpoint["epoch"])
    print("模型验证损失:", checkpoint["val_loss"]["total"])

    model = RoadsideVehicleDetector().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SplitHeightNetDataset(
        "data/splits/val.json",
        image_size=(576, 1024),
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=heightnet_collate_fn,
    )

    thresholds = [0.1, 0.2, 0.3, 0.5]
    match_distance = 2.0
    top_k = 50

    stats = {
        th: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "errors": [],
            "all_gt_errors": [],
        }
        for th in thresholds
    }
    
    for frame_no, batch in enumerate(loader, start=1):
        predictions = model(
            images=batch["image"].to(device),
            K=batch["intrinsics"].to(device),
            T=batch["extrinsics_raw"].to(device),
            ground_z=-30.0,
        )

        heatmap = (
            predictions["heatmap_logits"][0, 0]
            .sigmoid()
        )

        # 3x3局部极大值抑制
        pooled = F.max_pool2d(
            heatmap[None, None],
            kernel_size=3,
            stride=1,
            padding=1,
        )[0, 0]

        candidates = heatmap.masked_fill(
            heatmap != pooled, -1.0
        )

        scores, indices = candidates.flatten().topk(
            min(top_k, candidates.numel())
        )

        ny, nx = heatmap.shape
        rows = indices // nx
        cols = indices % nx

        offset = predictions["offset"][0]

        dx = offset[0, rows, cols].clamp(0, 1)
        dy = offset[1, rows, cols].clamp(0, 1)

        # BEV索引还原为世界坐标（单位：m）
        pred_x = 40.0 + cols.float() + dx
        pred_y = -32.0 + rows.float() + dy

        pred_xy = torch.stack(
            [pred_x, pred_y], dim=1
        ).cpu()

        scores = scores.cpu()
        gt_xy = batch["boxes_raw"][0][:, :2].cpu()

        for threshold in thresholds:
            selected = pred_xy[scores >= threshold]

            n_pred = len(selected)
            n_gt = len(gt_xy)

            matched_gt = set()
            tp = 0
            errors = []

            if n_pred > 0 and n_gt > 0:
                distances = torch.cdist(
                    selected, gt_xy
                )

                # 预测按置信度从高到低排列
                for i in range(n_pred):
                    d = distances[i].clone()

                    for j in matched_gt:
                        d[j] = float("inf")

                    j = int(d.argmin())
                    distance = float(d[j])

                    if distance <= match_distance:
                        matched_gt.add(j)
                        tp += 1
                        errors.append(distance)

            stats[threshold]["tp"] += tp
            stats[threshold]["fp"] += n_pred - tp
            stats[threshold]["fn"] += n_gt - tp
            stats[threshold]["errors"].extend(errors)
            # 全部GT车辆：漏检视为没有有限的定位误差
            stats[threshold]["all_gt_errors"].extend(
                errors + [float("inf")] * (n_gt - tp)
            )
        if frame_no % 20 == 0 or frame_no == len(loader):
            print(
                f"评估进度: {frame_no}/{len(loader)}",
                flush=True,
            )

    print("\n验证集车辆中心检测结果")
    print("匹配距离阈值:", match_distance, "m")
    print("单帧最大候选数:", top_k)

    for threshold in thresholds:
        s = stats[threshold]

        tp, fp, fn = s["tp"], s["fp"], s["fn"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        f1 = (
            2 * precision * recall
            / max(precision + recall, 1e-12)
        )

        mean_error = (
            sum(s["errors"]) / len(s["errors"])
            if s["errors"] else float("nan")
        )

        print(
            f"Score>={threshold:.1f} | "
            f"TP={tp} FP={fp} FN={fn} | "
            f"P={precision:.4f} "
            f"R={recall:.4f} "
            f"F1={f1:.4f} | "
            f"中心误差={mean_error:.4f}m"
        )
        p90_tp = percentile90(s["errors"])

        p90_all = percentile90(
            s["all_gt_errors"]
        )

        print(
            f"Score>={threshold:.1f} | "
            f"P90-TP={p90_tp:.4f} m | "
            f"P90-All-GT={p90_all:.4f} m"
        )

        if threshold == 0.1:
            total_gt = s["tp"] + s["fn"]

            for distance_limit in [0.5, 1.0, 2.0]:
                count = sum(
                    error <= distance_limit
                    for error in s["errors"]
                )

                rate = count / total_gt

                print(
                    f"定位误差≤{distance_limit:.1f}m: "
                    f"{count}/{total_gt} "
                    f"({rate * 100:.2f}%)"
                )


if __name__ == "__main__":
    main()
