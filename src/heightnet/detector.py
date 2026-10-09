
import torch
import torch.nn as nn


class BEVDetectionHead(nn.Module):
    """
    单类别、中心点式 BEV 车辆检测头。

    输入:
        bev: [B, 80, Ny, Nx]

    输出:
        heatmap_logits: [B, 1, Ny, Nx]
        offset:         [B, 2, Ny, Nx]
        log_sizes:      [B, 3, Ny, Nx]
        yaws:           [B, 2, Ny, Nx]
    """

    def __init__(self, in_channels=80, hidden_channels=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.heatmap = nn.Conv2d(
            hidden_channels, 1, kernel_size=1
        )

        self.offset = nn.Conv2d(
            hidden_channels, 2, kernel_size=1
        )

        self.log_sizes = nn.Conv2d(
            hidden_channels, 3, kernel_size=1
        )

        self.yaws = nn.Conv2d(
            hidden_channels, 2, kernel_size=1
        )

        # 降低初始车辆中心预测概率
        nn.init.constant_(self.heatmap.bias, -2.19)

    def forward(self, bev):
        x = self.shared(bev)

        return {
            "heatmap_logits": self.heatmap(x),
            "offset": self.offset(x),
            "log_sizes": self.log_sizes(x),
            "yaws": self.yaws(x),
        }
