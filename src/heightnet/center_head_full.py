
import torch
import torch.nn as nn


class SeparateHead(nn.Module):
    """
    CenterPoint 风格的独立预测分支。

    每个分支:
        Conv3x3 -> BN -> ReLU -> Conv3x3

    heatmap 最后一层 bias 初始化为 -2.19。
    """

    def __init__(
        self,
        in_channels=64,
        head_channels=64,
        num_classes=1,
        with_velocity=False,
    ):
        super().__init__()

        heads = {
            "heatmap": num_classes,
            "reg": 2,
            "height": 1,
            "dim": 3,
            "rot": 2,
        }

        if with_velocity:
            heads["vel"] = 2

        self.branches = nn.ModuleDict()

        for name, out_channels in heads.items():
            self.branches[name] = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    head_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(head_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
            )

        # CenterPoint 热力图分类头的典型初始化
        nn.init.constant_(
            self.branches["heatmap"][-1].bias,
            -2.19,
        )

    def forward(self, x):
        return {
            name: branch(x)
            for name, branch in self.branches.items()
        }


class CenterHeadFull(nn.Module):
    """
    面向 S3 车辆检测的 CenterPoint 风格预测头。

    输入:
        [B, 256, Ny, Nx]

    输出:
        {
            heatmap: [B, 1, Ny, Nx],
            reg:     [B, 2, Ny, Nx],
            height:  [B, 1, Ny, Nx],
            dim:     [B, 3, Ny, Nx],
            rot:     [B, 2, Ny, Nx],
            vel:     [B, 2, Ny, Nx],  # 可选
        }

    输出为原始预测值。
    heatmap 尚未经过 sigmoid。
    """

    def __init__(
        self,
        in_channels=256,
        shared_channels=64,
        num_classes=1,
        with_velocity=False,
    ):
        super().__init__()

        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                shared_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(shared_channels),
            nn.ReLU(inplace=True),
        )

        self.task_head = SeparateHead(
            in_channels=shared_channels,
            head_channels=64,
            num_classes=num_classes,
            with_velocity=with_velocity,
        )

    def forward(self, bev_features):
        if bev_features.ndim != 4:
            raise ValueError(
                "BEV features must be a 4D tensor"
            )

        if bev_features.shape[1] != 256:
            raise ValueError(
                "CenterHead expects 256 BEV channels"
            )

        shared = self.shared_conv(bev_features)
        return self.task_head(shared)
