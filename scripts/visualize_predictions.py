
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.heightnet.split_dataset import SplitHeightNetDataset
from src.heightnet.roadside_detector import RoadsideVehicleDetector
from src.heightnet.geometry import project_world_to_pixel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCENE = "S3S"
FRAME_ID = "300"

SCORE_THRESHOLD = 0.1
MATCH_DISTANCE = 2.0
TOP_K = 50

X_MIN = 40.0
Y_MIN = -32.0
SCALE = 8

# OpenCV使用BGR格式
COLOR_GT = (255, 255, 0)  # 青色：真实车辆
COLOR_TP = (0, 255, 0)    # 绿色：正确检测
COLOR_FP = (0, 0, 255)    # 红色：误检


@torch.inference_mode()
def main():
    # 1. 加载验证集
    dataset = SplitHeightNetDataset(
        "data/splits/val.json",
        image_size=(576, 1024),
    )

    index = next(
        (
            i
            for i, (scene, fid, _) in enumerate(dataset.records)
            if scene == SCENE and fid == FRAME_ID
        ),
        None,
    )

    if index is None:
        raise ValueError(
            f"{SCENE} 第{FRAME_ID}帧不在验证集中"
        )

    sample = dataset[index]

    # 2. 加载第5轮最优模型
    checkpoint = torch.load(
        "outputs/train_full/best.pth",
        map_location="cpu",
        weights_only=True,
    )

    model = RoadsideVehicleDetector().to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    print("加载模型Epoch:", checkpoint["epoch"])

    # 3. 模型推理
    predictions = model(
        images=sample["image"].unsqueeze(0).to(DEVICE),
        K=sample["intrinsics"].unsqueeze(0).to(DEVICE),
        T=sample["extrinsics_raw"].unsqueeze(0).to(DEVICE),
        ground_z=-30.0,
    )

    heat = predictions["heatmap_logits"][0, 0].sigmoid()

    # 4. 3x3局部极大值抑制
    pooled = F.max_pool2d(
        heat[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]

    candidates = heat.masked_fill(
        heat != pooled,
        -1.0,
    )

    scores, indices = candidates.flatten().topk(TOP_K)

    keep = scores >= SCORE_THRESHOLD
    scores = scores[keep]
    indices = indices[keep]

    ny, nx = heat.shape

    rows = indices // nx
    cols = indices % nx

    offsets = predictions["offset"][0]

    dx = offsets[0, rows, cols].clamp(0, 1)
    dy = offsets[1, rows, cols].clamp(0, 1)

    # 将BEV网格坐标还原为世界坐标
    pred_xy = torch.stack(
        [
            X_MIN + cols.float() + dx,
            Y_MIN + rows.float() + dy,
        ],
        dim=1,
    ).cpu()

    scores = scores.cpu()

    gt_xyz = sample["boxes_raw"][:, :3].cpu()
    gt_xy = gt_xyz[:, :2]

    # 5. 与真实车辆中心进行一对一匹配
    matched_gt = set()
    is_tp = []

    for point in pred_xy:
        if len(gt_xy) == 0:
            is_tp.append(False)
            continue

        distances = torch.linalg.norm(
            gt_xy - point.unsqueeze(0),
            dim=1,
        )

        for j in matched_gt:
            distances[j] = float("inf")

        j = int(distances.argmin())
        correct = float(distances[j]) <= MATCH_DISTANCE

        if correct:
            matched_gt.add(j)

        is_tp.append(correct)

    tp = sum(is_tp)
    fp = len(pred_xy) - tp
    fn = len(gt_xy) - tp

    print("场景:", SCENE)
    print("帧号:", FRAME_ID)
    print("真实车辆数量:", len(gt_xy))
    print("预测车辆数量:", len(pred_xy))
    print(f"TP={tp}, FP={fp}, FN={fn}")

    # 6. 绘制BEV热力图
    heat_np = (
        heat.cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    bev_image = cv2.applyColorMap(
        heat_np,
        cv2.COLORMAP_JET,
    )

    bev_image = cv2.resize(
        bev_image,
        (nx * SCALE, ny * SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    def bev_pixel(x, y):
        return (
            int(round((x - X_MIN) * SCALE)),
            int(round((y - Y_MIN) * SCALE)),
        )

    # GT：青色圆圈
    for point in gt_xy:
        x, y = point.tolist()

        cv2.circle(
            bev_image,
            bev_pixel(x, y),
            9,
            COLOR_GT,
            2,
        )

    # TP：绿色圆点；FP：红色叉号
    for point, correct in zip(pred_xy, is_tp):
        x, y = point.tolist()
        pixel = bev_pixel(x, y)

        if correct:
            cv2.circle(
                bev_image, pixel,
                5, COLOR_TP, -1,
            )
        else:
            cv2.drawMarker(
                bev_image,
                pixel,
                COLOR_FP,
                cv2.MARKER_TILTED_CROSS,
                12,
                2,
            )

    # 7. 投影回原相机图像
    image = sample["image"].permute(1, 2, 0).numpy()

    image = (
        image * 255
    ).clip(0, 255).astype(np.uint8)

    camera_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR,
    )

    pred_xyz = torch.cat(
        [
            pred_xy,
            torch.full(
                (len(pred_xy), 1),
                -30.0,
            ),
        ],
        dim=1,
    )

    all_xyz = torch.cat(
        [gt_xyz, pred_xyz],
        dim=0,
    )

    pixels, visible = project_world_to_pixel(
        all_xyz,
        sample["intrinsics"],
        sample["extrinsics_raw"],
    )

    for i, (pixel, valid) in enumerate(
        zip(pixels, visible)
    ):
        if not bool(valid):
            continue

        u, v = pixel.tolist()

        if not (
            0 <= u < camera_image.shape[1]
            and 0 <= v < camera_image.shape[0]
        ):
            continue

        p = (int(round(u)), int(round(v)))

        if i < len(gt_xyz):
            cv2.circle(
                camera_image,
                p,
                9,
                COLOR_GT,
                2,
            )
        else:
            correct = is_tp[i - len(gt_xyz)]

            if correct:
                cv2.circle(
                    camera_image,
                    p,
                    5,
                    COLOR_TP,
                    -1,
                )
            else:
                cv2.drawMarker(
                    camera_image,
                    p,
                    COLOR_FP,
                    cv2.MARKER_TILTED_CROSS,
                    12,
                    2,
                )

    # 8. 保存结果
    output_dir = Path(
        "outputs/vis/val_predictions"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bev_path = output_dir / (
        f"{SCENE}_{FRAME_ID}_bev.jpg"
    )

    camera_path = output_dir / (
        f"{SCENE}_{FRAME_ID}_camera.jpg"
    )

    cv2.imwrite(str(bev_path), bev_image)
    cv2.imwrite(str(camera_path), camera_image)

    print("\n可视化保存完成:")
    print("BEV图像:", bev_path)
    print("相机图像:", camera_path)


if __name__ == "__main__":
    main()
