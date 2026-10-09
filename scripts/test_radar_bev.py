from src.heightnet.split_dataset import SplitHeightNetDataset
from src.heightnet.radar_bev import RadarBEVEncoder



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



encoder=RadarBEVEncoder()



bev=encoder(
    radar_points
)



print(
    "Radar points:"
)


for p in radar_points:

    print(
        p.shape
    )



print(
    "Radar BEV:",
    bev.shape
)



print(
    "Non-zero cells:",
    (bev[:,0]>0).sum()
)



print(
    "Max density:",
    bev[:,0].max()
)