
import torch
import torch.nn as nn

from src.heightnet.backbone import ImageBackboneFPN
from src.heightnet.heightnet_full import HeightNetFull


class FullImageHeightEncoder(nn.Module):
    """
    ResNet50/101 + SECOND-FPN + 完整 HeightNet。

    输入:
        images:    [B, 3, H, W]
        K:         [B, 3, 3]，对应输入图像的内参
        world2cam: [B, 4, 4]

    输出:
        height_logits: [B, 90, H/16, W/16]
        height_prob:   [B, 90, H/16, W/16]
        context:       [B, 80, H/16, W/16]
    """

    def __init__(
        self,
        backbone_depth=50,
        backbone_pretrained=True,
        height_channels=90,
        context_channels=80,
    ):
        super().__init__()

        self.height_channels = height_channels
        self.context_channels = context_channels

        self.backbone = ImageBackboneFPN(
            output_channels=512,
            pretrained=backbone_pretrained,
            depth=backbone_depth,
            normalize_input=True,
        )

        self.heightnet = HeightNetFull(
            in_channels=512,
            mid_channels=512,
            height_channels=height_channels,
            context_channels=context_channels,
        )

    def forward(self, images, K, world2cam):
        features = self.backbone(images)

        output = self.heightnet(
            features, K, world2cam
        )

        height_logits, context = torch.split(
            output,
            [
                self.height_channels,
                self.context_channels,
            ],
            dim=1,
        )

        height_prob = torch.softmax(
            height_logits.float(),
            dim=1,
        )

        return {
            "height_logits": height_logits,
            "height_prob": height_prob,
            "context": context,
        }
