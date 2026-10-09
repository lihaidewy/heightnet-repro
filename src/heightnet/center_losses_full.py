
import torch
import torch.nn as nn


class CenterPointLoss(nn.Module):
    """
    单类别、无速度分支的 CenterPoint 风格损失。

    Heatmap: Gaussian Focal Loss
    Box:     masked weighted L1

    bbox_loss_weight=0.25 与官方配置对齐。
    """

    def __init__(self, bbox_loss_weight=0.25):
        super().__init__()
        self.bbox_loss_weight = bbox_loss_weight

        self.register_buffer(
            "code_weights",
            torch.ones(8, dtype=torch.float32)
        )

    def forward(self, predictions, targets):
        heat_logits = predictions["heatmap"].float()
        heat_target = targets["heatmap"].float()

        if heat_logits.shape != heat_target.shape:
            raise ValueError("Heatmap shape mismatch")

        pred_prob = heat_logits.sigmoid().clamp(
            1e-4, 1 - 1e-4
        )

        positive = (heat_target == 1).float()
        negative = (heat_target < 1).float()

        negative_weight = (
            1 - heat_target
        ).pow(4)

        positive_loss = (
            -(pred_prob.log())
            * (1 - pred_prob).pow(2)
            * positive
        )

        negative_loss = (
            -(1 - pred_prob).log()
            * pred_prob.pow(2)
            * negative_weight
            * negative
        )

        num_positive = positive.sum().clamp(min=1.0)

        heatmap_loss = (
            positive_loss.sum()
            + negative_loss.sum()
        ) / num_positive

        # 回归分支统一为8维编码
        encoded_box = torch.cat(
            [
                predictions["reg"],
                predictions["height"],
                predictions["dim"],
                predictions["rot"],
            ],
            dim=1,
        ).float()

        B, C, H, W = encoded_box.shape

        assert C == 8

        # [B,8,H,W] -> [B,H*W,8]
        encoded_box = (
            encoded_box.permute(0, 2, 3, 1)
            .contiguous()
            .view(B, -1, 8)
        )

        indices = targets["inds"]
        annotation = targets["anno"]
        valid = targets["mask"]

        gathered = encoded_box.gather(
            dim=1,
            index=indices.unsqueeze(-1).expand(
                -1, -1, 8
            )
        )

        error = torch.abs(
            gathered - annotation
        )

        weights = (
            valid.unsqueeze(-1).float()
            * self.code_weights.view(1, 1, 8)
        )

        num_objects = valid.sum().float().clamp(
            min=1.0
        )

        bbox_loss = (
            error * weights
        ).sum() / num_objects

        bbox_loss = (
            bbox_loss * self.bbox_loss_weight
        )

        total_loss = heatmap_loss + bbox_loss

        return {
            "total": total_loss,
            "heatmap": heatmap_loss,
            "bbox": bbox_loss,
        }
