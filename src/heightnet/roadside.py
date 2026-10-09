
import torch
import torch.nn as nn

from .backbone import ImageBackboneFPN
from .model import HeightNet, lift_features
from .height_bins import LIDHeightBins
from .geometry import build_height_frustum
from .bev_pool import BEVPool


class RoadsideHeightNet(nn.Module):
    """路侧 RGB -> BEV 特征的完整前向流程。"""

    def __init__(
        self,
        context_channels=80,
        num_height_bins=90,
        h_min=0.0,
        h_max=4.0,
        x_bound=(0.0, 160.0),
        y_bound=(-80.0, 80.0),
        bev_resolution=1.0,
    ):
        super().__init__()

        self.backbone = ImageBackboneFPN(
            output_channels=512,
            pretrained=False
        )

        self.heightnet = HeightNet(
            in_channels=512,
            mid_channels=128,
            context_channels=context_channels,
            num_height_bins=num_height_bins,
        )

        self.height_bins = LIDHeightBins(
            num_bins=num_height_bins,
            h_min=h_min,
            h_max=h_max,
        )

        self.bev_pool = BEVPool(
            x_bound=x_bound,
            y_bound=y_bound,
            resolution=bev_resolution,
        )

    def forward(
        self,
        images,
        K,
        T,
        ground_z=-30.0,
        return_debug=False,
    ):
        """
        images: [B, 3, H, W]
        K: [B, 3, 3]，已按图像尺寸缩放
        T: [B, 4, 4]，world -> camera
        ground_z: 世界坐标系参考地面高度
        """

        # 1. RGB -> 二维图像特征
        features_2d = self.backbone(images)

        # 2. HeightNet -> 语义与高度概率
        context, height_prob, height_logits = self.heightnet(
            features_2d,
            K,
            T,
            image_size=images.shape[-2:]
        )

        # 3. 特征沿高度维度展开
        features_3d = lift_features(
            context, height_prob
        )

        # 4. 每个高度假设对应的世界坐标
        xyz, valid = build_height_frustum(
            K=K,
            T=T,
            height_centers=self.height_bins.centers,
            feature_hw=context.shape[-2:],
            image_hw=images.shape[-2:],
            ground_z=ground_z,
        )

        # 5. 三维特征聚合为 BEV
        bev = self.bev_pool(
            features_3d, xyz, valid
        )

        if return_debug:
            return bev, {
                "context": context,
                "height_prob": height_prob,
                "height_logits": height_logits,
                "xyz": xyz,
                "valid": valid,
            }

        return bev
