import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .dataset import HeightNetDataset
from .radar_dataset import RadarLoader


class SplitHeightNetDataset(Dataset):
    """
    根据 train.json / val.json 加载 S3S、S3M 数据。

    不复制图像或标注文件，复用现有 HeightNetDataset。

    新增:
        radar_points:
            Tensor[N,4]

            [
                x,
                y,
                radial_velocity,
                snr
            ]
    """

    def __init__(
        self,
        split_file,
        image_size=(576, 1024),
    ):
        super().__init__()

        split_file = Path(split_file)

        with open(
            split_file,
            "r",
            encoding="utf-8"
        ) as f:

            records = json.load(f)


        if not records:

            raise ValueError(
                f"Empty split: {split_file}"
            )


        self.datasets = {}

        self.records = []


        # ==========================
        # Radar Loader
        # ==========================

        self.radar_loaders = {}


        radar_files = {

            "S3M":
                "data/S3M/radar.csv",

            "S3S":
                "data/S3S/radar.csv"

        }


        scenes = sorted(
            {
                record["scene"]
                for record in records
            }
        )


        for scene in scenes:

            if scene in radar_files:

                radar_path = Path(
                    radar_files[scene]
                )


                if radar_path.exists():

                    self.radar_loaders[scene] = RadarLoader(
                        str(radar_path),
                        radar_height=30.0,
                        yaw_deg=0.0
                    )


                    print(
                        f"[{scene}] Radar loaded:",
                        radar_path
                    )

                else:

                    print(
                        f"[Warning] Radar file missing:",
                        radar_path
                    )


        # ==========================
        # Image Dataset
        # ==========================

        frame_lookup = {}


        for scene in scenes:


            dataset = HeightNetDataset(
                f"data/{scene}",
                image_size=image_size,
            )


            self.datasets[scene] = dataset


            lookup = {}


            for index, sample_info in enumerate(
                dataset.samples
            ):

                frame_id = str(
                    sample_info[0]
                )


                if frame_id in lookup:

                    raise ValueError(
                        f"Duplicate frame: {scene}/{frame_id}"
                    )


                lookup[frame_id] = index


            frame_lookup[scene] = lookup



        # ==========================
        # Split index
        # ==========================

        for record in records:


            scene = record["scene"]

            frame_id = str(
                record["frame_id"]
            )


            if frame_id not in frame_lookup[scene]:

                raise FileNotFoundError(
                    f"Frame not found: {scene}/{frame_id}"
                )


            index = frame_lookup[scene][frame_id]


            self.records.append(
                (
                    scene,
                    frame_id,
                    index
                )
            )



        print(
            f"[{split_file.stem}] "
            f"Loaded samples: {len(self.records)}"
        )



    def __len__(self):

        return len(self.records)



    def __getitem__(
        self,
        index
    ):

        scene, frame_id, sample_index = self.records[index]


        sample = dict(
            self.datasets[scene][sample_index]
        )


        sample["scene"] = scene



        if str(sample["frame_id"]) != frame_id:

            raise RuntimeError(
                "Frame ID mismatch"
            )



        # ==========================
        # Radar
        # ==========================

        if scene in self.radar_loaders:


            radar_points = self.radar_loaders[scene].get(
                frame_id
            )


            if radar_points is None:

                radar_points = torch.zeros(
                    (0,4),
                    dtype=torch.float32
                )


        else:


            radar_points = torch.zeros(
                (0,4),
                dtype=torch.float32
            )



        sample["radar_points"] = radar_points



        return sample





def heightnet_collate_fn(batch):
    """
    支持不同帧具有不同数量的车辆3D Box。

    Radar:
        保留list
        因为不同frame雷达点数量不同。
    """


    return {

        "image":
            torch.stack(
                [
                    sample["image"]
                    for sample in batch
                ]
            ),


        "intrinsics":
            torch.stack(
                [
                    sample["intrinsics"]
                    for sample in batch
                ]
            ),


        "extrinsics_raw":
            torch.stack(
                [
                    sample["extrinsics_raw"]
                    for sample in batch
                ]
            ),


        "boxes_raw":
            [
                sample["boxes_raw"]
                for sample in batch
            ],



        "radar_points":
            [
                sample["radar_points"]
                for sample in batch
            ],



        "frame_id":
            [
                str(sample["frame_id"])
                for sample in batch
            ],



        "scene":
            [
                sample["scene"]
                for sample in batch
            ],

    }