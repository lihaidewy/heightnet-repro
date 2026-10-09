
import torch
import torch.nn as nn

from torchvision.models.resnet import BasicBlock


class BEVResidualStage(nn.Sequential):
    """
    ResNet18 风格的两层 BasicBlock 阶段。
    """

    def __init__(self, in_channels, out_channels, stride=1):
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            downsample = None

        super().__init__(
            BasicBlock(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
            ),
            BasicBlock(
                out_channels,
                out_channels,
                stride=1,
            ),
        )


class BEVSECONDFPN(nn.Module):
    """
    BEVHeight 官方配置对应的 SECOND-FPN。

    四级输入通道:
        [80, 160, 320, 640]

    上采样倍率:
        [1, 2, 4, 8]

    每级输出64通道，拼接后256通道。
    """

    def __init__(self):
        super().__init__()

        in_channels = [80, 160, 320, 640]
        upsample_strides = [1, 2, 4, 8]

        self.deblocks = nn.ModuleList()

        for channels, stride in zip(
            in_channels, upsample_strides
        ):
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    channels,
                    64,
                    kernel_size=stride,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    64,
                    eps=1e-3,
                    momentum=0.01,
                ),
                nn.ReLU(inplace=True),
            )

            self.deblocks.append(block)

    def forward(self, features):
        if len(features) != 4:
            raise ValueError(
                "BEVSECONDFPN expects four feature levels"
            )

        outputs = [
            block(feature)
            for block, feature in zip(
                self.deblocks, features
            )
        ]

        sizes = [x.shape[-2:] for x in outputs]

        if len(set(sizes)) != 1:
            raise RuntimeError(
                f"FPN spatial mismatch: {sizes}"
            )

        return torch.cat(outputs, dim=1)


class BEVEncoderFull(nn.Module):
    """
    BEVHeight 风格的 BEV ResNet18 + SECOND-FPN。

    输入:
        bev: [B, 80, H, W]

    输出:
        feature: [B, 256, H, W]

    使用:
        Stem stride=2
        无 MaxPool
        三个 BasicBlock 阶段
        四尺度 FPN
    """

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                80,
                160,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

        self.layer1 = BEVResidualStage(
            160, 160, stride=1
        )

        self.layer2 = BEVResidualStage(
            160, 320, stride=2
        )

        self.layer3 = BEVResidualStage(
            320, 640, stride=2
        )

        self.fpn = BEVSECONDFPN()

    def forward(self, bev):
        if bev.ndim != 4 or bev.shape[1] != 80:
            raise ValueError(
                "BEV input must have shape [B,80,H,W]"
            )

        h, w = bev.shape[-2:]

        if h % 8 != 0 or w % 8 != 0:
            raise ValueError(
                "BEV dimensions must be divisible by 8"
            )

        # 官方四尺度融合包括原始BEV特征
        f0 = bev

        x = self.stem(bev)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        output = self.fpn(
            [f0, f1, f2, f3]
        )

        expected_shape = (
            bev.shape[0], 256, h, w
        )

        if output.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected BEV shape: {output.shape}"
            )

        return output
