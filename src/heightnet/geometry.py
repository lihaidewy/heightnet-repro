
import torch


def project_world_to_pixel(points_world, K, T):
    """
    世界坐标 -> 图像像素

    points_world: [N, 3]
    K: [3, 3]
    T: [4, 4], world -> camera

    相机坐标约定：X前、Y右、Z上
    """
    R = T[:3, :3]
    t = T[:3, 3]

    points_cam = points_world @ R.T + t

    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    valid = X > 1e-6
    safe_X = torch.where(valid, X, torch.ones_like(X))

    # 相机坐标：X前、Y左、Z上
    u = K[0, 2] - K[0, 0] * Y / safe_X
    v = K[1, 2] - K[1, 1] * Z / safe_X

    pixels = torch.stack([u, v], dim=-1)

    return pixels, valid


def pixel_height_to_world(
    pixels, heights, K, T, ground_z
):
    """
    图像像素 + 离地高度 -> 世界坐标

    pixels: [N, 2]
    heights: [N]
    K: [3, 3]
    T: [4, 4], world -> camera
    ground_z: 世界坐标系中的参考地面高度

    返回:
        points_world: [N, 3]
        valid: [N]
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = pixels[:, 0]
    v = pixels[:, 1]

    # 1. 图像像素对应的相机射线
    # 相机坐标：X前、Y左、Z上
    rays_cam = torch.stack([
        torch.ones_like(u),
        (cx - u) / fx,
        (cy - v) / fy
    ], dim=-1)

    # 2. 相机射线 -> 世界坐标系
    R = T[:3, :3]
    t = T[:3, 3]

    rays_world = rays_cam @ R

    # 相机中心的世界坐标
    camera_center = -(R.T @ t)

    # 3. 射线与指定高度平面求交
    target_z = ground_z + heights
    ray_z = rays_world[:, 2]

    non_parallel = ray_z.abs() > 1e-6

    safe_ray_z = torch.where(
        non_parallel,
        ray_z,
        torch.ones_like(ray_z)
    )

    scale = (
        target_z - camera_center[2]
    ) / safe_ray_z

    valid = non_parallel & (scale > 0)

    points_world = (
        camera_center.unsqueeze(0)
        + scale.unsqueeze(-1) * rays_world
    )

    return points_world, valid

def build_height_frustum(
    K,
    T,
    height_centers,
    feature_hw,
    image_hw,
    ground_z=-30.0
):
    """
    将特征图的每个空间位置、每个高度分箱
    映射到世界坐标系。

    参数:
        K: [B, 3, 3]
        T: [B, 4, 4]，world -> camera
        height_centers: [D]
        feature_hw: (Hf, Wf)
        image_hw: (H, W)
        ground_z: 参考地面在世界坐标系中的 Z

    返回:
        xyz: [B, D, Hf, Wf, 3]
        valid: [B, D, Hf, Wf]
    """
    B = K.shape[0]
    Hf, Wf = feature_hw
    H, W = image_hw

    device = K.device
    dtype = K.dtype

    height_centers = height_centers.to(
        device=device, dtype=dtype
    )

    D = height_centers.numel()

    # 1. 构建特征位置对应的原图像素坐标
    # 使用每个下采样单元的近似中心
    u = (
        (torch.arange(Wf, device=device, dtype=dtype) + 0.5)
        * (W / Wf) - 0.5
    )

    v = (
        (torch.arange(Hf, device=device, dtype=dtype) + 0.5)
        * (H / Hf) - 0.5
    )

    vv, uu = torch.meshgrid(v, u, indexing="ij")

    pixels = torch.stack([uu, vv], dim=-1)

    # 2. 对每个像素位置展开 D 个高度假设
    pixels = (
        pixels.reshape(1, Hf * Wf, 2)
        .expand(D, -1, -1)
        .reshape(-1, 2)
    )

    heights = (
        height_centers.reshape(D, 1)
        .expand(D, Hf * Wf)
        .reshape(-1)
    )

    xyz_list = []
    valid_list = []

    # 3. 每张图像独立进行几何投影
    for b in range(B):
        points, mask = pixel_height_to_world(
            pixels=pixels,
            heights=heights,
            K=K[b],
            T=T[b],
            ground_z=ground_z
        )

        xyz_list.append(
            points.reshape(D, Hf, Wf, 3)
        )

        valid_list.append(
            mask.reshape(D, Hf, Wf)
        )

    xyz = torch.stack(xyz_list, dim=0)
    valid = torch.stack(valid_list, dim=0)

    return xyz, valid