import pandas as pd
import os

from .radar_utils import radar_polar_to_bev



class RadarLoader:


    def __init__(
        self,
        csv_file,
        radar_height=30.0,
        yaw_deg=0.0
    ):


        self.data=pd.read_csv(
            csv_file
        )


        self.radar_height=radar_height

        self.yaw_deg=yaw_deg



        self.frames={}



        for frame,group in self.data.groupby(
            "Frame"
        ):

            points=radar_polar_to_bev(
                group["Range"].values,
                group["Angle"].values,
                group["Speed"].values,
                group["SNR"].values,
                radar_height,
                yaw_deg
            )


            self.frames[
                str(frame)
            ]=points



    def get(
        self,
        frame_id
    ):

        frame_id=str(frame_id)


        if frame_id in self.frames:

            return self.frames[frame_id]


        else:

            return None