import torch

from src.heightnet.radar_bev import RadarBEVEncoder
from src.heightnet.radar_fusion import RadarBEVFusion

from src.heightnet.split_dataset import SplitHeightNetDataset



dataset=SplitHeightNetDataset(
    "data/splits/eval_S3M.json",
    image_size=(576,1024)
)



sample1=dataset[0]

sample2=dataset[1]



radar_points=[

    sample1["radar_points"],

    sample2["radar_points"]

]



# Radar BEV

radar_encoder=RadarBEVEncoder()


radar_bev=radar_encoder(
    radar_points
)



print(
    "Radar BEV:",
    radar_bev.shape
)



# 模拟 Image BEV

image_bev=torch.randn(
    2,
    256,
    80,
    128
)



print(
    "Image BEV:",
    image_bev.shape
)



fusion=RadarBEVFusion()



fused=fusion(
    image_bev,
    radar_bev
)



print(
    "Fused BEV:",
    fused.shape
)



# gradient test

loss=fused.mean()

loss.backward()



print(
    "Fusion gradient:",
    fusion.fusion[0].weight.grad is not None
)