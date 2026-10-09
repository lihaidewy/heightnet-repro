
import torch
import torch.nn as nn


class LIDHeightBins(nn.Module):
    """
    论文公式 (2-1)：Linear-Increased Discretization。

    使用 N+1 个边界构建 N 个高度区间。
    每个区间的中心高度作为几何投影的代表值。
    """

    def __init__(self, num_bins=90, h_min=0.0, h_max=4.0):
        super().__init__()

        if num_bins <= 0:
            raise ValueError("num_bins must be positive")
        if h_max <= h_min:
            raise ValueError("h_max must be greater than h_min")

        self.num_bins = num_bins

        i = torch.arange(num_bins + 1, dtype=torch.float32)

        edges = h_min + (h_max - h_min) * (
            i * (i + 1)
        ) / (num_bins * (num_bins + 1))

        centers = (edges[:-1] + edges[1:]) / 2

        self.register_buffer("edges", edges)
        self.register_buffer("centers", centers)

    def height_to_index(self, height):
        """连续高度转换成离散区间编号。"""
        index = torch.bucketize(
            height.contiguous(),
            self.edges[1:-1].contiguous()
        )

        return index.clamp(0, self.num_bins - 1)

    def forward(self, height_logits):
        """
        输入:
            [B, N, H, W] 高度分类 logits

        输出:
            probability: [B, N, H, W]
            expected_height: [B, H, W]
        """
        if height_logits.shape[1] != self.num_bins:
            raise ValueError("Height channel count mismatch")

        probability = height_logits.softmax(dim=1)

        centers = self.centers.to(
            dtype=probability.dtype
        )

        expected_height = (
            probability * centers.view(1, -1, 1, 1)
        ).sum(dim=1)

        return probability, expected_height
