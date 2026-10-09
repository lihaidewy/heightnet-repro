
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import (
    SplitHeightNetDataset,
    heightnet_collate_fn,
)
from src.heightnet.targets import BEVTargetEncoder
from src.heightnet.roadside_detector import RoadsideVehicleDetector
from src.heightnet.losses import BEVDetectionLoss


LOSS_NAMES = ("total", "heatmap", "offset", "size", "yaw")


def run_epoch(
    model,
    loader,
    criterion,
    encoder,
    device,
    optimizer=None,
    max_batches=0,
):
    training = optimizer is not None
    model.train(training)

    totals = {name: 0.0 for name in LOSS_NAMES}
    total_samples = 0

    batch_limit = (
        min(max_batches, len(loader))
        if max_batches > 0 else len(loader)
    )

    for step, batch in enumerate(loader, start=1):
        if step > batch_limit:
            break

        images = batch["image"].to(
            device, non_blocking=True
        )
        K = batch["intrinsics"].to(
            device, non_blocking=True
        )
        T = batch["extrinsics_raw"].to(
            device, non_blocking=True
        )

        targets = [
            encoder(boxes)
            for boxes in batch["boxes_raw"]
        ]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(
                images=images,
                K=K,
                T=T,
                ground_z=-30.0,
            )

            losses = criterion(
                predictions,
                targets,
            )

            loss = losses["total"]

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at batch {step}"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

        batch_size = images.shape[0]
        total_samples += batch_size

        for name in LOSS_NAMES:
            totals[name] += (
                losses[name].detach().item()
                * batch_size
            )

        if (
            step == 1
            or step % 20 == 0
            or step == batch_limit
        ):
            mode = "Train" if training else "Val"

            print(
                f"{mode} "
                f"[{step}/{batch_limit}] "
                f"Loss={loss.item():.4f}",
                flush=True,
            )

    if total_samples == 0:
        raise RuntimeError("No samples processed")

    return {
        name: value / total_samples
        for name, value in totals.items()
    }


def save_checkpoint(
    path, model, optimizer, epoch,
    train_loss, val_loss, args
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)

    # 0 表示使用全部 Batch
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train_full",
    )

    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")

    if args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("max batch counts cannot be negative")

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("设备:", device)
    print("输出目录:", output_dir)

    # -------------------------
    # 1. 数据集与 DataLoader
    # -------------------------
    train_dataset = SplitHeightNetDataset(
        "data/splits/train.json",
        image_size=(576, 1024),
    )

    val_dataset = SplitHeightNetDataset(
        "data/splits/val.json",
        image_size=(576, 1024),
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": heightnet_collate_fn,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    print("训练帧数:", len(train_dataset))
    print("验证帧数:", len(val_dataset))
    print("训练Batch:", len(train_loader))
    print("验证Batch:", len(val_loader))

    # -------------------------
    # 2. 检测监督
    # -------------------------
    encoder = BEVTargetEncoder(
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0,
    )

    # -------------------------
    # 3. 模型和优化器
    # -------------------------
    model = RoadsideVehicleDetector().to(device)
    criterion = BEVDetectionLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # 从头训练，不加载单帧过拟合权重
    best_val_loss = float("inf")

    csv_path = output_dir / "training_log.csv"

    fieldnames = (
        ["epoch"]
        + [f"train_{name}" for name in LOSS_NAMES]
        + [f"val_{name}" for name in LOSS_NAMES]
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -------------------------
    # 4. 多帧训练
    # -------------------------
    with open(
        csv_path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            print(
                f"\n========== Epoch "
                f"{epoch}/{args.epochs} ==========",
                flush=True,
            )

            train_losses = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                encoder=encoder,
                device=device,
                optimizer=optimizer,
                max_batches=args.max_train_batches,
            )

            # 验证时禁用梯度，不更新模型参数
            with torch.no_grad():
                val_losses = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    encoder=encoder,
                    device=device,
                    optimizer=None,
                    max_batches=args.max_val_batches,
                )

            print(
                f"Epoch {epoch} | "
                f"Train={train_losses['total']:.6f} | "
                f"Val={val_losses['total']:.6f}",
                flush=True,
            )

            # -------------------------
            # 5. 保存 CSV 日志
            # -------------------------
            row = {"epoch": epoch}

            for name in LOSS_NAMES:
                row[f"train_{name}"] = train_losses[name]
                row[f"val_{name}"] = val_losses[name]

            writer.writerow(row)
            file.flush()

            # -------------------------
            # 6. 保存最新模型
            # -------------------------
            save_checkpoint(
                output_dir / "last.pth",
                model,
                optimizer,
                epoch,
                train_losses,
                val_losses,
                args,
            )

            # -------------------------
            # 7. 保存验证损失最优模型
            # -------------------------
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]

                save_checkpoint(
                    output_dir / "best.pth",
                    model,
                    optimizer,
                    epoch,
                    train_losses,
                    val_losses,
                    args,
                )

                print(
                    f"更新最优模型: "
                    f"Val={best_val_loss:.6f}",
                    flush=True,
                )

    print("\n训练结束")
    print("最优验证损失:", best_val_loss)
    print("训练日志:", csv_path)
    print("最新权重:", output_dir / "last.pth")
    print("最优权重:", output_dir / "best.pth")

    if device.type == "cuda":
        peak_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )
        print(f"GPU峰值分配显存: {peak_mb:.2f} MB")


if __name__ == "__main__":
    main()
