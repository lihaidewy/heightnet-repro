import torch
import torch.nn as nn


class RadarBEVFusion(nn.Module):
    """
    Image BEV + Radar BEV fusion


    Input:

        image_bev:

            [B,256,H,W]


        radar_bev:

            [B,4,H,W]


    Output:

        fused_bev:

            [B,256,H,W]

    """


    def __init__(
        self,
        image_channels=256,
        radar_channels=4,
        out_channels=256
    ):

        super().__init__()


        self.fusion = nn.Sequential(

            nn.Conv2d(
                image_channels + radar_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )

        )


    def forward(
        self,
        image_bev,
        radar_bev
    ):


        if image_bev.ndim != 4:

            raise ValueError(
                f"image_bev must be [B,C,H,W], got {image_bev.shape}"
            )


        if radar_bev.ndim != 4:

            raise ValueError(
                f"radar_bev must be [B,C,H,W], got {radar_bev.shape}"
            )


        if image_bev.shape[0] != radar_bev.shape[0]:

            raise ValueError(
                "Batch size mismatch"
            )


        if image_bev.shape[-2:] != radar_bev.shape[-2:]:

            raise ValueError(
                f"BEV size mismatch: "
                f"{image_bev.shape[-2:]} vs {radar_bev.shape[-2:]}"
            )


        fused=torch.cat(
            [
                image_bev,
                radar_bev
            ],
            dim=1
        )


        fused=self.fusion(
            fused
        )


        return fused