
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """多尺度空洞空间金字塔池化。"""

    def __init__(self, channels=128):
        super().__init__()

        branch_channels = channels // 4

        def conv_branch(kernel_size, dilation):
            padding = dilation if kernel_size == 3 else 0
            return nn.Sequential(
                nn.Conv2d(
                    channels, branch_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=False
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            )

        self.branch1 = conv_branch(1, 1)
        self.branch2 = conv_branch(3, 6)
        self.branch3 = conv_branch(3, 12)
        self.branch4 = conv_branch(3, 18)

        # 全局上下文分支
        # 不在 1x1 全局特征上使用 BatchNorm，
        # 避免 batch_size=1 时的训练错误。
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels, branch_channels, 1
            ),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                branch_channels * 5,
                channels, 1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[-2:]

        global_feat = self.global_pool(x)
        global_feat = F.interpolate(
            global_feat,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        features = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            global_feat
        ]

        return self.project(
            torch.cat(features, dim=1)
        )
