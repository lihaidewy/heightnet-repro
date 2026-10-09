
import torch
import torch.nn as nn

from src.heightnet.full_image_encoder import FullImageHeightEncoder
from src.heightnet.geometry_full import HeightGeometryFull
from src.heightnet.bev_pool_full import VoxelPoolingFull


class FullBEVFeatureExtractor(nn.Module):
    """
    完整路侧 RGB -> BEV 特征提取网络。

    流程：
        RGB
          -> ResNet50 + SECOND-FPN
          -> Camera-Aware HeightNet
          -> Height Probability × Context
          -> Height-to-World Geometry
          -> Voxel Pooling
          -> BEV Feature

    输入:
        images:    [B, 3, H, W]
        K:         [B, 3, 3]
        world2cam: [B, 4, 4]

    输出:
        bev:         [B, 80, 80, 128]
        height_logits
        height_prob
        geometry_valid
    """

    def __init__(
        self,
        backbone_depth=50,
        backbone_pretrained=True,
        height_channels=90,
        context_channels=80,
    ):
        super().__init__()

        self.image_encoder = FullImageHeightEncoder(
            backbone_depth=backbone_depth,
            backbone_pretrained=backbone_pretrained,
            height_channels=height_channels,
            context_channels=context_channels,
        )

        self.geometry = HeightGeometryFull(
            num_bins=height_channels,
            height_min=0.0,
            height_max=4.0,
            height_method="lid",
            pixel_mode="official",
            max_forward=300.0,
        )

        self.bev_pool = VoxelPoolingFull(
            x_bound=(40.0, 168.0, 1.0),
            y_bound=(-32.0, 48.0, 1.0),
            z_bound=(-30.0, -26.0, 4.0),
            chunk_bins=12,
        )

    def forward(
        self,
        images,
        K,
        world2cam,
        ground_z=-30.0,
    ):
        # 1. 图像特征 + 高度预测
        predictions = self.image_encoder(
            images, K, world2cam
        )

        context = predictions["context"]
        height_prob = predictions["height_prob"]

        # 2. 高度空间几何映射
        # K 和 T 是固定标定值，
        # 无须通过几何坐标计算梯度。
        with torch.no_grad():
            mapping = self.geometry(
                K=K,
                world2cam=world2cam,
                image_hw=images.shape[-2:],
                feature_hw=context.shape[-2:],
                ground_z=ground_z,
            )

        # 3. 带高度概率的特征体素池化
        bev = self.bev_pool(
            context=context,
            height_prob=height_prob,
            xyz=mapping["xyz"],
            valid=mapping["valid"],
        )

        return {
            "bev": bev,
            "height_logits": predictions["height_logits"],
            "height_prob": height_prob,
            "geometry_valid": mapping["valid"],
        }
