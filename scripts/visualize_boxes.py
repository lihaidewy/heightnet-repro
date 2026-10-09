
import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCENE = "S3S"
FRAME = sys.argv[1] if len(sys.argv) > 1 else "257"

root = Path("data") / SCENE
image_path = root / "images" / f"{FRAME}.png"
label_path = root / "labels" / f"{FRAME}.json"
output_dir = Path("outputs/vis") / SCENE
output_dir.mkdir(parents=True, exist_ok=True)

image = cv2.imread(str(image_path))
if image is None:
    raise FileNotFoundError(image_path)

with open(label_path, "r", encoding="utf-8") as f:
    ann = json.load(f)

intr = ann["intrinsics"]
K = np.array([
    [intr["fx"], 0, intr["cx"]],
    [0, intr["fy"], intr["cy"]],
    [0, 0, 1],
], dtype=np.float64)

# 已通过 translation_cam 验证：
# T 将原始标注坐标转换到相机坐标
T = np.asarray(
    ann["extrinsics_matrix_4x4"],
    dtype=np.float64
)

# 3D 包围框的 12 条边
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

COLORS = [
    (0, 255, 0),
    (255, 128, 0),
    (0, 0, 255),
    (255, 0, 255)
]


def get_box_corners(obj, center_mode):
    """根据 3D Box 生成世界坐标系下的 8 个角点。

    暂定:
        size = [length, width, height]
        yaw 绕世界 Z 轴旋转
    """
    x, y, z = obj["translation"]
    length, width, height = obj["size"]
    yaw = obj["rotation"]

    if center_mode == "bottom":
        z_values = [0.0, height]
    else:
        z_values = [-height / 2, height / 2]

    # 每一层按相同顺序排列四个角点
    xy = [
        [ length / 2,  width / 2],
        [ length / 2, -width / 2],
        [-length / 2, -width / 2],
        [-length / 2,  width / 2]
    ]

    local_corners = np.array(
        [[px, py, pz]
         for pz in z_values
         for px, py in xy],
        dtype=np.float64
    )

    c = np.cos(yaw)
    s = np.sin(yaw)

    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    rotated = local_corners @ R.T
    translated = rotated + np.array([x, y, z])

    return translated


def project_points(points_world, y_direction):
    """尝试 X 前、Z 上的相机投影模型。"""

    homogeneous = np.concatenate(
        [points_world, np.ones((len(points_world), 1))],
        axis=1
    )

    points_cam = (T @ homogeneous.T).T[:, :3]

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid = X > 1e-4
    pixels = np.zeros((len(X), 2), dtype=np.float64)

    if y_direction == "right":
        sign = 1.0
    else:
        sign = -1.0

    pixels[valid, 0] = (
        intr["cx"] + sign * intr["fx"] * Y[valid] / X[valid]
    )
    pixels[valid, 1] = (
        intr["cy"] - intr["fy"] * Z[valid] / X[valid]
    )

    return pixels, valid


def draw_boxes(y_direction, center_mode):
    canvas = image.copy()

    for idx, obj in enumerate(ann["objects"]):
        corners = get_box_corners(obj, center_mode)
        pixels, valid = project_points(corners, y_direction)
        color = COLORS[idx % len(COLORS)]

        for a, b in EDGES:
            if not (valid[a] and valid[b]):
                continue

            p1 = tuple(np.round(pixels[a]).astype(int))
            p2 = tuple(np.round(pixels[b]).astype(int))

            cv2.line(canvas, p1, p2, color, 5)

        # 绘制标注坐标本身对应的投影位置
        center_world = np.asarray(
            [obj["translation"]], dtype=np.float64
        )
        center_px, center_valid = project_points(
            center_world, y_direction
        )

        if center_valid[0]:
            p = tuple(np.round(center_px[0]).astype(int))
            cv2.circle(canvas, p, 10, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"{idx}: {obj['category']}",
                (p[0] + 12, p[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 3
            )

    cv2.putText(
        canvas,
        f"Y={y_direction} / Origin={center_mode}",
        (60, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 255), 4
    )

    preview = cv2.resize(
        canvas, (1280, 720),
        interpolation=cv2.INTER_AREA
    )

    name = f"{FRAME}_Y_{y_direction}_{center_mode}.jpg"
    output_path = output_dir / name

    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Cannot save: {output_path}")

    print("已保存:", output_path)


if __name__ == "__main__":
    print("Frame:", FRAME)
    print("Objects:", len(ann["objects"]))

    for y_direction in ["right", "left"]:
        for center_mode in ["bottom", "center"]:
            draw_boxes(y_direction, center_mode)
