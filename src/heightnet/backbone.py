
import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    resnet101,
    ResNet50_Weights,
    ResNet101_Weights,
)


class SECONDFPN(nn.Module):
    """
    四尺度特征融合，与 BEVHeight 官方 SECOND-FPN
    的尺度变换方式对齐。

    输入:
        C2: [B, 256,  H/4,  W/4]
        C3: [B, 512,  H/8,  W/8]
        C4: [B, 1024, H/16, W/16]
        C5: [B, 2048, H/32, W/32]

    输出:
        [B, 512, H/16, W/16]
    """

    def __init__(self):
        super().__init__()

        in_channels = [256, 512, 1024, 2048]
        out_channels = [128, 128, 128, 128]

        self.deblocks = nn.ModuleList()

        # 前两级向下采样
        # 后两级保持或向上采样
        strides = [4, 2, 1, 2]

        for i, stride in enumerate(strides):

            if i < 2:
                conv = nn.Conv2d(
                    in_channels[i],
                    out_channels[i],
                    kernel_size=stride,
                    stride=stride,
                    bias=False,
                )
            else:
                conv = nn.ConvTranspose2d(
                    in_channels[i],
                    out_channels[i],
                    kernel_size=stride,
                    stride=stride,
                    bias=False,
                )

            block = nn.Sequential(
                conv,
                nn.BatchNorm2d(
                    out_channels[i],
                    eps=1e-3,
                    momentum=0.01,
                ),
                nn.ReLU(inplace=True),
            )

            self.deblocks.append(block)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(
                "SECONDFPN expects four feature levels"
            )

        outputs = [
            block(feature)
            for block, feature in zip(
                self.deblocks, features
            )
        ]

        spatial_sizes = [
            output.shape[-2:]
            for output in outputs
        ]

        if len(set(spatial_sizes)) != 1:
            raise RuntimeError(
                f"FPN spatial mismatch: {spatial_sizes}"
            )

        return torch.cat(outputs, dim=1)


class ImageBackboneFPN(nn.Module):
    """
    ResNet50 / ResNet101 + SECOND-FPN。

    输入:
        [B, 3, H, W]，RGB，数值范围 [0,1]

    输出:
        [B, 512, H/16, W/16]
    """

    def __init__(
        self,
        output_channels=512,
        pretrained=False,
        depth=50,
        normalize_input=True,
    ):
        super().__init__()

        if output_channels != 512:
            raise ValueError(
                "Official-style SECOND-FPN outputs "
                "512 channels"
            )

        if depth == 50:
            weights = (
                ResNet50_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            net = resnet50(weights=weights)

        elif depth == 101:
            weights = (
                ResNet101_Weights.IMAGENET1K_V1
                if pretrained else None
            )
            net = resnet101(weights=weights)

        else:
            raise ValueError(
                "Supported depths: 50, 101"
            )

        self.depth = depth
        self.normalize_input = normalize_input

        # ImageNet 归一化参数
        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406]
            ).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225]
            ).view(1, 3, 1, 1),
        )

        # ResNet Stem
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
        )

        # 四级残差特征
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # 完整四尺度 SECOND-FPN
        self.fpn = SECONDFPN()

    def forward(self, images):
        h, w = images.shape[-2:]

        if h % 32 != 0 or w % 32 != 0:
            raise ValueError(
                "Image dimensions must be divisible by 32"
            )

        x = images

        if self.normalize_input:
            x = (
                x - self.image_mean
            ) / self.image_std

        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        fused = self.fpn(
            [c2, c3, c4, c5]
        )

        return fused
