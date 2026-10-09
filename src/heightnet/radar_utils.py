import numpy as np
import torch


def radar_polar_to_bev(
    ranges,
    angles,
    speeds,
    snrs,
    radar_height=30.0,
    yaw_deg=0.0
):
    """
    Radar polar coordinates -> BEV coordinates

    Args:
        ranges:
            slant range (m)

        angles:
            horizontal azimuth degree

        speeds:
            radial velocity

        snrs:
            radar intensity


    Returns:

        Tensor [N,4]

        x
        y
        velocity
        snr
    """

    ranges=np.asarray(ranges)
    angles=np.asarray(angles)
    speeds=np.asarray(speeds)
    snrs=np.asarray(snrs)


    # remove invalid points

    valid=ranges > radar_height


    ranges=ranges[valid]
    angles=angles[valid]
    speeds=speeds[valid]
    snrs=snrs[valid]



    # slant -> ground distance

    ground=np.sqrt(
        ranges**2 -
        radar_height**2
    )



    theta=np.deg2rad(
        angles
    )



    x=ground*np.cos(theta)

    y=ground*np.sin(theta)



    # yaw rotation

    yaw=np.deg2rad(
        yaw_deg
    )


    cos_y=np.cos(yaw)

    sin_y=np.sin(yaw)


    x_new=(
        cos_y*x -
        sin_y*y
    )


    y_new=(
        sin_y*x +
        cos_y*y
    )



    points=np.stack(
        [
            x_new,
            y_new,
            speeds,
            snrs
        ],
        axis=1
    )


    return torch.tensor(
        points,
        dtype=torch.float32
    )