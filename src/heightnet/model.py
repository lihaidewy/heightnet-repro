
import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from .aspp import ASPP

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(x + self.net(x))


class HeightNet(nn.Module):
    def __init__(
        self,
        in_channels=512,
        mid_channels=128,
        context_channels=80,
        num_height_bins=90,
    ):
        super().__init__()

        self.num_height_bins = num_height_bins

        # 输入二维特征通道调整
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 分别对相机内参与外参进行编码
        self.intrinsic_mlp = nn.Sequential(
            nn.Linear(4, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
        )

        self.extrinsic_mlp = nn.Sequential(
            nn.Linear(12, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
        )

        # 两个分支各自使用相机感知通道权重
        self.context_gate = nn.Linear(mid_channels, mid_channels)
        self.height_gate = nn.Linear(mid_channels, mid_channels)

        # Context Branch
        self.context_branch = nn.Sequential(
            ResBlock(mid_channels),
            nn.Conv2d(mid_channels, context_channels, 1),
        )

        # Height Branch
        self.height_res = nn.Sequential(
            ResBlock(mid_channels),
            ResBlock(mid_channels),
            ResBlock(mid_channels),
        )
        self.height_aspp = ASPP(mid_channels)

        # DCN：偏移由网络自行预测
        self.offset_conv = nn.Conv2d(
            mid_channels, 18, 3, padding=1
        )

        self.dcn = DeformConv2d(
            mid_channels, mid_channels,
            kernel_size=3, padding=1
        )

        self.height_head = nn.Conv2d(
            mid_channels, num_height_bins, 1
        )

        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

    def forward(self, features, K, T, image_size):
        """
        features: [B, C, H', W']
        K:        [B, 3, 3]  调整图像尺寸后的内参
        T:        [B, 4, 4]  标注系到相机系外参
        image_size: (H, W)
        """
        batch_size = features.shape[0]
        image_h, image_w = image_size

        x = self.reduce_conv(features)

        # 归一化相机内参
        kvec = torch.stack([
            K[:, 0, 0] / image_w,
            K[:, 1, 1] / image_h,
            K[:, 0, 2] / image_w,
            K[:, 1, 2] / image_h,
        ], dim=1)

        tvec = T[:, :3, :4].reshape(batch_size, 12)

        cam = (
            self.intrinsic_mlp(kvec)
            + self.extrinsic_mlp(tvec)
        )

        # Context Branch
        context_weight = torch.sigmoid(
            self.context_gate(cam)
        ).unsqueeze(-1).unsqueeze(-1)

        context = self.context_branch(x * context_weight)

        # Height Branch
        height_weight = torch.sigmoid(
            self.height_gate(cam)
        ).unsqueeze(-1).unsqueeze(-1)

        height_feat = self.height_res(x * height_weight)

        # 多尺度特征提取
        height_feat = self.height_aspp(height_feat)

        offsets = self.offset_conv(height_feat)
        height_feat = self.dcn(height_feat, offsets)

        height_logits = self.height_head(height_feat)
        height_prob = torch.softmax(height_logits, dim=1)

        return context, height_prob, height_logits


def lift_features(context, height_prob):
    """公式(2-3)：按高度概率提升二维特征。"""
    return context.unsqueeze(2) * height_prob.unsqueeze(1)
