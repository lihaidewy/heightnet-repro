
from pathlib import Path

import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.targets import BEVTargetEncoder
from src.heightnet.roadside_detector import RoadsideVehicleDetector
from src.heightnet.losses import BEVDetectionLoss


def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 1. 固定一帧真实图像
    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    index = next(
        i for i, (fid, _, _) in enumerate(dataset.samples)
        if str(fid) == "257"
    )

    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(device)
    K = sample["intrinsics"].unsqueeze(0).to(device)
    T = sample["extrinsics_raw"].unsqueeze(0).to(device)

    # 2. 构造检测监督
    encoder = BEVTargetEncoder(
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0,
    )

    target = encoder(sample["boxes_raw"])

    print("帧号:", sample["frame_id"])
    print("车辆数量:", len(target["indices"]))
    print("训练设备:", device)

    # 3. 模型、损失、优化器
    model = RoadsideVehicleDetector().to(device)
    criterion = BEVDetectionLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # 记录训练前的参数，用于验证权重是否更新
    weight_before = (
        model.roadside.heightnet.height_head.weight
        .detach().clone()
    )

    model.train()

    num_steps = 60
    history = []

    print("\n开始单帧过拟合实验\n")

    for step in range(1, num_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        # 完整前向传播
        predictions = model(
            images=image,
            K=K,
            T=T,
            ground_z=-30.0,
        )

        # 真实3D Box检测损失
        losses = criterion(predictions, [target])
        total_loss = losses["total"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(
                f"Step {step}: loss is not finite"
            )

        # 反向传播与参数更新
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        history.append(float(total_loss.item()))

        if step == 1 or step % 10 == 0:
            # 观察真实车辆中心位置的预测概率
            with torch.no_grad():
                probability = (
                    predictions["heatmap_logits"]
                    .sigmoid().reshape(-1)
                )

                indices = target["indices"].to(device)

                center_score = (
                    probability[indices].mean().item()
                )

            print(
                f"Step {step:03d}/{num_steps} | "
                f"Total={total_loss.item():.4f} | "
                f"HM={losses['heatmap'].item():.4f} | "
                f"Offset={losses['offset'].item():.4f} | "
                f"Size={losses['size'].item():.4f} | "
                f"Yaw={losses['yaw'].item():.4f} | "
                f"GT中心概率={center_score:.4f}"
            )

    # 4. 检查 HeightNet 参数是否真实更新
    weight_after = (
        model.roadside.heightnet.height_head.weight
        .detach()
    )

    weight_change = (
        weight_after - weight_before
    ).abs().max().item()

    print("\n训练结果:")
    print(f"初始损失: {history[0]:.6f}")
    print(f"最终损失: {history[-1]:.6f}")
    print(f"Height Head权重最大变化: {weight_change:.8f}")

    assert weight_change > 0, "HeightNet参数没有更新"

    # 5. 保存模型
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "overfit_s3s_257.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": num_steps,
            "loss_history": history,
            "frame_id": "257",
            "scene": "S3S",
        },
        save_path,
    )

    print("模型已保存:", save_path)

    if device.type == "cuda":
        print(
            "GPU峰值分配显存(MB):",
            round(
                torch.cuda.max_memory_allocated() / 1024**2,
                2
            )
        )

    print("\n单帧训练流程执行完成")


if __name__ == "__main__":
    main()
