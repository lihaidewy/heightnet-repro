
import torch.nn as nn

from .roadside import RoadsideHeightNet
from .detector import BEVDetectionHead


class RoadsideVehicleDetector(nn.Module):
    """
    完整路侧车辆检测模型：

    RGB -> Backbone + FPN -> HeightNet
        -> 3D Projection -> BEV Pooling
        -> BEV Detection Head
    """

    def __init__(self):
        super().__init__()

        # 与 BEVTargetEncoder 使用相同的空间范围
        self.roadside = RoadsideHeightNet(
            context_channels=80,
            num_height_bins=90,
            h_min=0.0,
            h_max=4.0,
            x_bound=(40.0, 168.0),
            y_bound=(-32.0, 48.0),
            bev_resolution=1.0,
        )

        self.detector = BEVDetectionHead(
            in_channels=80,
            hidden_channels=64,
        )

    def forward(
        self,
        images,
        K,
        T,
        ground_z=-30.0,
        return_debug=False,
    ):
        if return_debug:
            bev, debug = self.roadside(
                images=images,
                K=K,
                T=T,
                ground_z=ground_z,
                return_debug=True,
            )
        else:
            bev = self.roadside(
                images=images,
                K=K,
                T=T,
                ground_z=ground_z,
            )

        predictions = self.detector(bev)

        if return_debug:
            debug["bev"] = bev
            return predictions, debug

        return predictions
