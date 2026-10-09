
import torch
import torch.nn as nn

from src.heightnet.full_bev_extractor import (
    FullBEVFeatureExtractor,
)
from src.heightnet.bev_encoder_full import BEVEncoderFull
from src.heightnet.center_head_full import CenterHeadFull
from src.heightnet.center_targets_full import (
    CenterTargetGenerator,
)
from src.heightnet.center_losses_full import (
    CenterPointLoss,
)


class FullBEVHeightDetector(nn.Module):
    """
    S3 数据适配版完整 BEVHeight 检测网络。

    RGB
     -> ResNet50 + Image SECOND-FPN
     -> Camera-Aware HeightNet
     -> Height Geometry + Voxel Pooling
     -> BEV ResNet18 + BEV SECOND-FPN
     -> CenterPoint Detection Head

    输入:
        images:    [B,3,H,W]
        K:         [B,3,3]
        world2cam: [B,4,4]

    检测输出:
        heatmap, reg, height, dim, rot
    """

    def __init__(
        self,
        backbone_pretrained=True,
        backbone_depth=50,
        ground_z=-30.0,
    ):
        super().__init__()

        self.ground_z = float(ground_z)

        # 1. RGB -> BEV，输出80通道
        self.bev_extractor = FullBEVFeatureExtractor(
            backbone_depth=backbone_depth,
            backbone_pretrained=backbone_pretrained,
            height_channels=90,
            context_channels=80,
        )

        # 2. BEV特征编码，80 -> 256通道
        self.bev_encoder = BEVEncoderFull()

        # 3. CenterPoint风格检测头
        self.detection_head = CenterHeadFull(
            in_channels=256,
            shared_channels=64,
            num_classes=1,
            with_velocity=False,
        )

        # 4. 监督目标
        self.target_generator = CenterTargetGenerator(
            x_bound=(40.0, 168.0, 1.0),
            y_bound=(-32.0, 48.0, 1.0),
            ground_z=self.ground_z,
            max_objs=500,
            gaussian_overlap=0.1,
            min_radius=2,
        )

        # 5. 检测损失
        self.criterion = CenterPointLoss(
            bbox_loss_weight=0.25,
        )

    def forward(self, images, K, world2cam):
        # RGB -> BEV
        extracted = self.bev_extractor(
            images=images,
            K=K,
            world2cam=world2cam,
            ground_z=self.ground_z,
        )

        # BEV特征编码
        bev_features = self.bev_encoder(
            extracted["bev"]
        )

        # 车辆检测
        predictions = self.detection_head(
            bev_features
        )

        return {
            "predictions": predictions,
            "bev": extracted["bev"],
            "bev_features": bev_features,
            "height_prob": extracted["height_prob"],
        }

    def compute_loss(self, predictions, boxes_list):
        """
        boxes_list: 长度B的列表。

        每个元素为 [N,7] Tensor：
        x,y,z_bottom,length,width,height,yaw
        """
        device = predictions["heatmap"].device

        targets = self.target_generator(
            boxes_list,
            device=device,
        )

        losses = self.criterion(
            predictions,
            targets,
        )

        return losses, targets
