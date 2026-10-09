
import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVDetectionLoss(nn.Module):
    """
    单类别中心点式 BEV 检测损失。

    targets 为长度为 B 的列表，
    每个元素来自 BEVTargetEncoder。
    """

    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        logits = predictions["heatmap_logits"]

        B, _, Ny, Nx = logits.shape

        if len(targets) != B:
            raise ValueError(
                "Target batch size does not match predictions"
            )

        device = logits.device

        # -------------------------
        # 1. Heatmap Focal Loss
        # -------------------------
        gt_heatmap = torch.stack(
            [t["heatmap"] for t in targets],
            dim=0
        ).to(device=device, dtype=logits.dtype)

        if gt_heatmap.shape != logits.shape:
            raise ValueError("Heatmap shape mismatch")

        prob = logits.sigmoid().clamp(
            min=1e-4,
            max=1.0 - 1e-4
        )

        positive = gt_heatmap.eq(1.0)
        negative = gt_heatmap.lt(1.0)

        negative_weight = (
            1.0 - gt_heatmap
        ).pow(4)

        pos_loss = (
            -torch.log(prob)
            * (1.0 - prob).pow(2)
            * positive
        )

        neg_loss = (
            -torch.log(1.0 - prob)
            * prob.pow(2)
            * negative_weight
            * negative
        )

        num_positive = positive.sum().clamp(min=1)

        loss_heatmap = (
            pos_loss.sum() + neg_loss.sum()
        ) / num_positive

        # -------------------------
        # 2. 仅在真实中心位置回归
        # -------------------------
        loss_offset = logits.new_zeros(())
        loss_size = logits.new_zeros(())
        loss_yaw = logits.new_zeros(())

        total_objects = 0

        for b in range(B):
            target = targets[b]

            indices = target["indices"].to(
                device=device,
                dtype=torch.long
            )

            if indices.numel() == 0:
                continue

            if (indices < 0).any() or (indices >= Ny * Nx).any():
                raise ValueError("Target index is outside BEV")

            total_objects += indices.numel()

            pairs = [
                ("offset", "offsets"),
                ("log_sizes", "log_sizes"),
                ("yaws", "yaws"),
            ]

            for prediction_key, target_key in pairs:
                pred_map = predictions[prediction_key][b]

                # [C, Ny, Nx] -> [Ny*Nx, C]
                flat = pred_map.permute(
                    1, 2, 0
                ).reshape(-1, pred_map.shape[0])

                selected = flat.index_select(
                    0, indices
                )

                gt = target[target_key].to(
                    device=device,
                    dtype=selected.dtype
                )

                if selected.shape != gt.shape:
                    raise ValueError(
                        f"Shape mismatch: {prediction_key}"
                    )

                reg_loss = F.smooth_l1_loss(
                    selected,
                    gt,
                    reduction="sum"
                )

                if prediction_key == "offset":
                    loss_offset = loss_offset + reg_loss
                elif prediction_key == "log_sizes":
                    loss_size = loss_size + reg_loss
                else:
                    loss_yaw = loss_yaw + reg_loss

        denominator = max(total_objects, 1)

        loss_offset = loss_offset / denominator
        loss_size = loss_size / denominator
        loss_yaw = loss_yaw / denominator

        # -------------------------
        # 3. 总损失
        # -------------------------
        total_loss = (
            loss_heatmap
            + loss_offset
            + 0.2 * loss_size
            + 0.2 * loss_yaw
        )

        return {
            "total": total_loss,
            "heatmap": loss_heatmap,
            "offset": loss_offset,
            "size": loss_size,
            "yaw": loss_yaw,
        }
