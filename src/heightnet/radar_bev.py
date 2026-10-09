import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarBEVEncoder(nn.Module):
    """
    Radar points -> BEV feature


    Input:

        radar_points:

        List[Tensor]

        each:
            [N,4]

        columns:

            0: x
            1: y
            2: radial velocity
            3: SNR



    Output:

        radar_bev:

        [B,4,80,128]


    BEV channels:

        0:
            point density

        1:
            mean velocity

        2:
            mean SNR

        3:
            mean range

    """


    def __init__(
        self,
        x_bound=(0.0,160.0),
        y_bound=(-80.0,80.0),
        resolution=1.0,
        output_size=(80,128)
    ):

        super().__init__()


        self.x_min=x_bound[0]
        self.x_max=x_bound[1]

        self.y_min=y_bound[0]
        self.y_max=y_bound[1]


        self.resolution=resolution


        self.nx=int(
            (self.x_max-self.x_min)
            /
            resolution
        )

        self.ny=int(
            (self.y_max-self.y_min)
            /
            resolution
        )


        self.output_size=output_size



    def forward(
        self,
        radar_points
    ):


        batch_bev=[]


        device=radar_points[0].device



        for points in radar_points:


            bev=torch.zeros(
                (
                    4,
                    self.ny,
                    self.nx
                ),
                dtype=torch.float32,
                device=device
            )


            if points.numel()==0:

                batch_bev.append(
                    bev
                )

                continue



            points=points.to(device)



            x=points[:,0]
            y=points[:,1]

            velocity=points[:,2]
            snr=points[:,3]


            distance=torch.sqrt(
                x*x+y*y
            )



            ix=torch.floor(
                (x-self.x_min)
                /
                self.resolution
            ).long()


            iy=torch.floor(
                (y-self.y_min)
                /
                self.resolution
            ).long()



            valid=(

                (ix>=0)
                &
                (ix<self.nx)
                &
                (iy>=0)
                &
                (iy<self.ny)

            )


            ix=ix[valid]
            iy=iy[valid]

            velocity=velocity[valid]
            snr=snr[valid]
            distance=distance[valid]



            if ix.numel()==0:

                batch_bev.append(
                    bev
                )

                continue



            count=torch.zeros(
                (self.ny,self.nx),
                device=device
            )

            vel_sum=torch.zeros_like(
                count
            )

            snr_sum=torch.zeros_like(
                count
            )

            range_sum=torch.zeros_like(
                count
            )


            ones=torch.ones_like(
                velocity
            )


            count.index_put_(
                (iy,ix),
                ones,
                accumulate=True
            )


            vel_sum.index_put_(
                (iy,ix),
                velocity,
                accumulate=True
            )


            snr_sum.index_put_(
                (iy,ix),
                snr,
                accumulate=True
            )


            range_sum.index_put_(
                (iy,ix),
                distance,
                accumulate=True
            )



            mask=count>0



            bev[0]=count


            bev[1][mask]=(
                vel_sum[mask]
                /
                count[mask]
            )


            bev[2][mask]=(
                snr_sum[mask]
                /
                count[mask]
            )


            bev[3][mask]=(
                range_sum[mask]
                /
                count[mask]
            )



            batch_bev.append(
                bev
            )



        bev=torch.stack(
            batch_bev,
            dim=0
        )


        # 
        # [B,4,160,160]
        #
        # -> 
        #
        # [B,4,80,128]
        #

        bev=F.interpolate(
            bev,
            size=self.output_size,
            mode="bilinear",
            align_corners=False
        )


        return bev