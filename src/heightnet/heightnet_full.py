
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import BasicBlock
from mmcv.ops import DeformConv2dPack


def build_camera_condition(K, world2cam, ida=None, bda=None):
    """
    构造与官方 HeightNet 字段顺序一致的 27 维条件向量。

    K:         [B, 3, 3]，已缩放到网络输入分辨率
    world2cam: [B, 4, 4]，S3 JSON 原始外参
    ida:       [B, 4, 4]，图像增强变换
    bda:       [B, 4, 4]，BEV 增强变换

    返回: [B, 27]
    """
    B = K.shape[0]

    if K.shape != (B, 3, 3):
        raise ValueError(f"Invalid K shape: {K.shape}")

    if world2cam.shape != (B, 4, 4):
        raise ValueError(
            f"Invalid world2cam shape: {world2cam.shape}"
        )

    identity = torch.eye(
        4, dtype=K.dtype, device=K.device
    ).unsqueeze(0).expand(B, -1, -1)

    if ida is None:
        ida = identity

    if bda is None:
        bda = identity

    # S3: world -> camera
    # 条件编码需要 camera -> world
    cam2world = torch.linalg.inv(world2cam)

    intrinsics = torch.stack(
        [
            K[:, 0, 0],
            K[:, 1, 1],
            K[:, 0, 2],
            K[:, 1, 2],
        ],
        dim=-1,
    )

    image_aug = torch.stack(
        [
            ida[:, 0, 0],
            ida[:, 0, 1],
            ida[:, 0, 3],
            ida[:, 1, 0],
            ida[:, 1, 1],
            ida[:, 1, 3],
        ],
        dim=-1,
    )

    bev_aug = torch.stack(
        [
            bda[:, 0, 0],
            bda[:, 0, 1],
            bda[:, 1, 0],
            bda[:, 1, 1],
            bda[:, 2, 2],
        ],
        dim=-1,
    )

    extrinsics = cam2world[:, :3, :4].reshape(B, 12)

    condition = torch.cat(
        [intrinsics, image_aug, bev_aug, extrinsics],
        dim=-1,
    )

    assert condition.shape == (B, 27)
    return condition


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class SELayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_reduce = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv_expand = nn.Conv2d(
            channels, channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, condition):
        gate = self.conv_reduce(condition)
        gate = self.relu(gate)
        gate = self.conv_expand(gate)
        return features * self.sigmoid(gate)


class ASPPBranch(nn.Sequential):
    def __init__(self, channels, dilation):
        kernel_size = 1 if dilation == 1 else 3
        padding = 0 if dilation == 1 else dilation

        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class ASPP(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branches = nn.ModuleList(
            [
                ASPPBranch(channels, dilation)
                for dilation in [1, 6, 12, 18]
            ]
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels, channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                channels * 5,
                channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]

        pooled = self.global_pool(x)
        pooled = F.interpolate(
            pooled,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        outputs.append(pooled)
        return self.project(torch.cat(outputs, dim=1))


class HeightNetFull(nn.Module):
    """
    高规格 Camera-Aware HeightNet。

    输入:
        features: [B, 512, H, W]
        K:        [B, 3, 3]
        world2cam:[B, 4, 4]

    输出:
        [B, 90 + 80, H, W]

    前 90 通道为高度 logits；
    后 80 通道为上下文特征。
    """

    def __init__(
        self,
        in_channels=512,
        mid_channels=512,
        context_channels=80,
        height_channels=90,
    ):
        super().__init__()

        self.height_channels = height_channels
        self.context_channels = context_channels

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.camera_bn = nn.BatchNorm1d(27)

        self.context_mlp = MLP(
            27, mid_channels, mid_channels
        )
        self.height_mlp = MLP(
            27, mid_channels, mid_channels
        )

        self.context_se = SELayer(mid_channels)
        self.height_se = SELayer(mid_channels)

        self.context_conv = nn.Conv2d(
            mid_channels,
            context_channels,
            kernel_size=1,
        )

        self.height_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels),
            DeformConv2dPack(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                deform_groups=1,
                im2col_step=128,
            ),
        )

        self.height_layer = nn.Conv2d(
            mid_channels,
            height_channels,
            kernel_size=1,
        )

    def forward(self, features, K, world2cam, ida=None, bda=None):
        camera_condition = build_camera_condition(
            K, world2cam, ida=ida, bda=bda
        )

        camera_condition = self.camera_bn(
            camera_condition
        )

        x = self.reduce_conv(features)

        context_gate = self.context_mlp(
            camera_condition
        )[..., None, None]

        height_gate = self.height_mlp(
            camera_condition
        )[..., None, None]

        context = self.context_se(x, context_gate)
        context = self.context_conv(context)

        height = self.height_se(x, height_gate)
        height = self.height_conv(height)
        height_logits = self.height_layer(height)

        return torch.cat(
            [height_logits, context],
            dim=1,
        )
