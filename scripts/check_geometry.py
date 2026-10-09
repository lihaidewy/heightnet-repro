
import json
from pathlib import Path
import numpy as np
from PIL import Image

scene = Path("data/S3S")

for frame_id in ["194", "257"]:
    print(f"\n{'=' * 50}")
    print(f"检查帧: {frame_id}")
    print("=" * 50)

    label_path = scene / "labels" / f"{frame_id}.json"
    image_path = scene / "images" / f"{frame_id}.png"

    if not label_path.exists() or not image_path.exists():
        print("图像或标注不存在")
        continue

    with open(label_path, encoding="utf-8") as f:
        ann = json.load(f)

    with Image.open(image_path) as img:
        width, height = img.size

    K = ann["intrinsics"]
    T = np.asarray(
        ann["extrinsics_matrix_4x4"],
        dtype=np.float64
    )

    print("图像尺寸:", width, height)
    print("外参矩阵:\n", T)

    for obj in ann["objects"][:3]:
        p_world = np.array(
            [*obj["translation"], 1.0],
            dtype=np.float64
        )

        p_direct = (T @ p_world)[:3]
        p_inverse = (np.linalg.inv(T) @ p_world)[:3]

        print("\n目标:", obj["category"])
        print("原始坐标:", p_world[:3])
        print("T直接变换:", p_direct)
        print("T逆向变换:", p_inverse)

        if "translation_cam" in obj:
            given = np.asarray(
                obj["translation_cam"],
                dtype=np.float64
            )

            print("标注中的相机坐标:", given)
            print(
                "直接变换误差:",
                np.linalg.norm(p_direct - given)
            )
            print(
                "逆向变换误差:",
                np.linalg.norm(p_inverse - given)
            )

        # 暂以 T 直接变换结果检查不同相机轴约定
        x, y, z = p_direct
        fx, fy = K["fx"], K["fy"]
        cx, cy = K["cx"], K["cy"]

        candidates = []

        if x > 1e-6:
            candidates.extend([
                (
                    "X前/Y右/Z上",
                    cx + fx * y / x,
                    cy - fy * z / x
                ),
                (
                    "X前/Y左/Z上",
                    cx - fx * y / x,
                    cy - fy * z / x
                )
            ])

        if z > 1e-6:
            candidates.append((
                "OpenCV Z前",
                cx + fx * x / z,
                cy + fy * y / z
            ))

        print("\n候选像素位置:")
        for name, u, v in candidates:
            inside = (
                0 <= u < width
                and 0 <= v < height
            )
            print(
                f"{name}: "
                f"u={u:.1f}, v={v:.1f}, "
                f"位于图像内={inside}"
            )

print("\n检查完成")
