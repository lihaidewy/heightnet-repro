
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import SplitHeightNetDataset
from src.heightnet.full_detector import FullBEVHeightDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full BEVHeight S3 training"
    )

    parser.add_argument(
        "--train-split",
        default="data/splits/train.json",
    )
    parser.add_argument(
        "--val-split",
        default="data/splits/val.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/train_heightnet_full",
    )

    parser.add_argument(
        "--image-height", type=int, default=576
    )
    parser.add_argument(
        "--image-width", type=int, default=1024
    )
    parser.add_argument(
        "--batch-size", type=int, default=2
    )
    parser.add_argument(
        "--num-workers", type=int, default=2
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--amp",
        choices=["fp32", "bf16"],
        default="fp32",
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    # 用于短批次 Smoke Test。
    # 0 表示整个数据集。
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
        "--log-interval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def collate_fn(samples):
    """
    boxes_raw 长度不同，必须保留为列表。
    不对车辆框进行 padding 或丢弃。
    """
    return {
        "image": torch.stack([
            torch.as_tensor(s["image"])
            for s in samples
        ]),

        "intrinsics": torch.stack([
            torch.as_tensor(s["intrinsics"])
            for s in samples
        ]),

        "extrinsics_raw": torch.stack([
            torch.as_tensor(s["extrinsics_raw"])
            for s in samples
        ]),

        "boxes_raw": [
            torch.as_tensor(s["boxes_raw"])
            for s in samples
        ],
    }


def split_sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def verify_splits(args):
    with open(args.train_split, encoding="utf-8") as f:
        train = json.load(f)

    with open(args.val_split, encoding="utf-8") as f:
        val = json.load(f)

    train_ids = {
        (item["scene"], str(item["frame_id"]))
        for item in train
    }

    val_ids = {
        (item["scene"], str(item["frame_id"]))
        for item in val
    }

    assert len(train) == 395
    assert len(val) == 114
    assert len(train_ids) == len(train)
    assert len(val_ids) == len(val)
    assert not (train_ids & val_ids), (
        "Train/Val contains overlapping frames"
    )

    print("训练集:", len(train))
    print("验证集:", len(val))
    print("训练/验证交集: 0")

    return {
        "train": split_sha256(args.train_split),
        "val": split_sha256(args.val_split),
    }


def make_loader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    generator,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=generator,
        worker_init_fn=seed_worker,
    )


def move_batch(batch, device):
    return (
        batch["image"].to(
            device, dtype=torch.float32,
            non_blocking=True,
        ),
        batch["intrinsics"].to(
            device, dtype=torch.float32,
            non_blocking=True,
        ),
        batch["extrinsics_raw"].to(
            device, dtype=torch.float32,
            non_blocking=True,
        ),
        [
            boxes.to(
                device, dtype=torch.float32,
                non_blocking=True,
            )
            for boxes in batch["boxes_raw"]
        ],
    )


def build_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
    min_lr_ratio,
):
    warmup_steps = min(
        warmup_steps,
        max(1, total_steps // 10),
    )

    def lr_scale(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps

        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps)
        )

        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

        return (
            min_lr_ratio
            + (1.0 - min_lr_ratio) * cosine
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_scale
    )

    return scheduler, warmup_steps


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    best_val,
    args,
    split_hashes,
):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_val": best_val,
        "args": vars(args),
        "split_hashes": split_hashes,
        "rng_state": get_rng_state(),
    }

    # 先保存临时文件，再原子替换。
    tmp_path = str(path) + ".tmp"

    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scheduler=None,
    amp="fp32",
    grad_clip=5.0,
    max_batches=0,
    log_interval=20,
    epoch=1,
):
    training = optimizer is not None

    model.train(training)

    total = 0.0
    total_heatmap = 0.0
    total_bbox = 0.0
    total_samples = 0
    total_objects = 0

    start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        images, K, T, boxes_list = move_batch(
            batch, device
        )

        batch_size = images.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(amp == "bf16"),
            ):
                outputs = model(
                    images, K, T
                )

                losses, targets = model.compute_loss(
                    outputs["predictions"],
                    boxes_list,
                )

                loss = losses["total"]

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss in epoch {epoch}, "
                    f"batch {batch_idx}"
                )

            expected_gt = sum(
                len(boxes) for boxes in boxes_list
            )

            actual_gt = int(
                targets["mask"].sum().item()
            )

            if actual_gt != expected_gt:
                raise RuntimeError(
                    f"GT mismatch: expected={expected_gt}, "
                    f"encoded={actual_gt}"
                )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip,
                    error_if_nonfinite=True,
                )

                optimizer.step()
                scheduler.step()

        total += loss.item() * batch_size
        total_heatmap += (
            losses["heatmap"].item() * batch_size
        )
        total_bbox += (
            losses["bbox"].item() * batch_size
        )

        total_samples += batch_size
        total_objects += actual_gt

        if training and (
            (batch_idx + 1) % log_interval == 0
        ):
            print(
                f"Epoch {epoch} | "
                f"batch {batch_idx + 1} | "
                f"loss {loss.item():.4f} | "
                f"lr {optimizer.param_groups[0]['lr']:.8f}",
                flush=True,
            )

        del outputs, losses, targets, loss

    if total_samples == 0:
        raise RuntimeError("No batches processed")

    torch.cuda.synchronize()

    return {
        "loss": total / total_samples,
        "heatmap": total_heatmap / total_samples,
        "bbox": total_bbox / total_samples,
        "samples": total_samples,
        "objects": total_objects,
        "seconds": time.perf_counter() - start,
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    if args.batch_size < 2:
        raise ValueError(
            "Full HeightNet training requires batch_size >= 2 "
            "because camera_bn uses BatchNorm1d(27)"
        )

    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")

    if args.image_height % 32 or args.image_width % 32:
        raise ValueError(
            "Image dimensions must be divisible by 32"
        )

    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    set_seed(args.seed)

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("AMP:", args.amp)

    if args.amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "BF16 not supported on current device"
            )

    split_hashes = verify_splits(args)

    image_size = (
        args.image_height,
        args.image_width,
    )

    train_dataset = SplitHeightNetDataset(
        args.train_split,
        image_size=image_size,
    )

    val_dataset = SplitHeightNetDataset(
        args.val_split,
        image_size=image_size,
    )

    assert len(train_dataset) == 395
    assert len(val_dataset) == 114

    train_generator = torch.Generator()
    val_generator = torch.Generator().manual_seed(
        args.seed + 100000
    )

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        train_generator,
    )

    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        val_generator,
    )

    steps_per_epoch = len(train_loader)

    if args.max_train_batches:
        steps_per_epoch = min(
            steps_per_epoch,
            args.max_train_batches,
        )

    total_steps = steps_per_epoch * args.epochs

    print("每轮训练批次数:", steps_per_epoch)
    print("总优化步数:", total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    csv_path = output_dir / "training_log.csv"
    config_path = output_dir / "config.json"

    if not args.resume and last_path.exists():
        raise RuntimeError(
            f"Existing checkpoint: {last_path}. "
            "Use --resume or a different output directory."
        )

    model = FullBEVHeightDetector(
        backbone_pretrained=True,
        backbone_depth=50,
        ground_z=-30.0,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=False,
    )

    scheduler, warmup_steps = build_scheduler(
        optimizer,
        total_steps,
        args.warmup_steps,
        args.min_lr_ratio,
    )

    print("Warmup steps:", warmup_steps)

    num_params = sum(
        p.numel() for p in model.parameters()
    )

    print("参数量:", f"{num_params:,}")

    start_epoch = 1
    best_val = float("inf")

    resume_keys = [
        "train_split",
        "val_split",
        "image_height",
        "image_width",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "warmup_steps",
        "min_lr_ratio",
        "max_train_batches",
        "max_val_batches",
        "amp",
        "seed",
    ]

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )

        if checkpoint["split_hashes"] != split_hashes:
            raise RuntimeError(
                "Train/Val split changed since checkpoint"
            )

        previous = checkpoint["args"]

        for key in resume_keys:
            if previous[key] != getattr(args, key):
                raise RuntimeError(
                    f"Resume config mismatch: {key}"
                )

        model.load_state_dict(
            checkpoint["model"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        best_val = float(checkpoint["best_val"])
        start_epoch = int(checkpoint["epoch"]) + 1

        restore_rng_state(
            checkpoint["rng_state"]
        )

        print(
            "恢复训练，下一轮 Epoch:",
            start_epoch,
        )

        if not csv_path.exists():
            raise RuntimeError(
                "CSV log missing for resumed run"
            )

    else:
        with open(
            config_path, "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "args": vars(args),
                    "split_hashes": split_hashes,
                    "warmup_steps_effective": warmup_steps,
                    "model_parameters": num_params,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(
            csv_path, "w", newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(f)

            writer.writerow([
                "epoch",
                "train_loss",
                "train_heatmap",
                "train_bbox",
                "val_loss",
                "val_heatmap",
                "val_bbox",
                "train_samples",
                "val_samples",
                "train_objects",
                "val_objects",
                "lr",
                "seconds",
                "peak_memory_mb",
            ])

    if start_epoch > args.epochs:
        print("训练已完成，无需继续。")
        return

    for epoch in range(
        start_epoch, args.epochs + 1
    ):
        print(
            f"\n========== Epoch "
            f"{epoch}/{args.epochs} ==========",
            flush=True,
        )

        torch.cuda.reset_peak_memory_stats(device)

        epoch_start = time.perf_counter()

        # 每轮使用由种子和轮次决定的采样顺序。
        # 断点续训时不依赖上一轮DataLoader状态。
        train_generator.manual_seed(
            args.seed + epoch
        )

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            amp=args.amp,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
            log_interval=args.log_interval,
            epoch=epoch,
        )

        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            scheduler=None,
            amp=args.amp,
            max_batches=args.max_val_batches,
            epoch=epoch,
        )

        elapsed = time.perf_counter() - epoch_start

        peak_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch} | "
            f"train={train_stats['loss']:.6f} | "
            f"val={val_stats['loss']:.6f} | "
            f"lr={lr:.8f} | "
            f"time={elapsed:.1f}s | "
            f"peak={peak_mb:.1f}MB",
            flush=True,
        )

        improved = (
            val_stats["loss"] < best_val
        )

        if improved:
            best_val = val_stats["loss"]

        with open(
            csv_path, "a", newline="",
            encoding="utf-8"
        ) as f:
            csv.writer(f).writerow([
                epoch,
                train_stats["loss"],
                train_stats["heatmap"],
                train_stats["bbox"],
                val_stats["loss"],
                val_stats["heatmap"],
                val_stats["bbox"],
                train_stats["samples"],
                val_stats["samples"],
                train_stats["objects"],
                val_stats["objects"],
                lr,
                elapsed,
                peak_mb,
            ])

        save_checkpoint(
            last_path,
            epoch,
            model,
            optimizer,
            scheduler,
            best_val,
            args,
            split_hashes,
        )

        if improved:
            save_checkpoint(
                best_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_val,
                args,
                split_hashes,
            )

            print(
                f"Best updated: "
                f"Epoch {epoch}, "
                f"val_loss={best_val:.6f}",
                flush=True,
            )

    print("\n高规格 BEVHeight 训练完成")
    print("最优验证损失:", best_val)
    print("最优权重:", best_path)


if __name__ == "__main__":
    main()
