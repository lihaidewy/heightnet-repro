
import torch
import torch.nn as nn


def create_height_samples(
    num_bins=90,
    height_min=0.0,
    height_max=4.0,
    method="lid",
):
    """
    高度采样位置，单位 m。

    lid:
        使用非均匀 LID 分箱的区间中心。

    did:
        使用 BEVHeight 官方代码中的
        幂指数 alpha=1.5 的采样方式。
    """
    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    if height_max <= height_min:
        raise ValueError("Invalid height range")

    i = torch.arange(
        num_bins, dtype=torch.float32
    )

    if method == "lid":
        edges_i = torch.arange(
            num_bins + 1, dtype=torch.float32
        )
        edges = (
            height_min
            + (height_max - height_min)
            * edges_i * (edges_i + 1)
            / (num_bins * (num_bins + 1))
        )
        samples = (edges[:-1] + edges[1:]) * 0.5

    elif method == "did":
        samples = (
            height_min
            + (height_max - height_min)
            * (i / num_bins).pow(1.5)
        )

    else:
        raise ValueError(
            "method must be 'lid' or 'did'"
        )

    return samples


class HeightGeometryFull(nn.Module):
    """
    将图像特征网格和高度分箱转换到世界坐标系。

    相机坐标约定:
        X 前、Y 左、Z 上

    输入:
        K:         [B,3,3]
        world2cam: [B,4,4]
        image_hw:  (H_img,W_img)
        feature_hw:(H_feat,W_feat)
        ground_z:  世界坐标地面高度
        ida:       可选 [B,4,4] 图像增强矩阵

    输出:
        xyz:   [B,D,H_feat,W_feat,3]
        valid: [B,D,H_feat,W_feat]
    """

    def __init__(
        self,
        num_bins=90,
        height_min=0.0,
        height_max=4.0,
        height_method="lid",
        pixel_mode="official",
        max_forward=300.0,
    ):
        super().__init__()

        self.pixel_mode = pixel_mode
        self.max_forward = max_forward

        samples = create_height_samples(
            num_bins,
            height_min,
            height_max,
            height_method,
        )

        self.register_buffer(
            "height_samples", samples
        )

    def create_pixel_grid(
        self, image_hw, feature_hw, device, dtype
    ):
        image_h, image_w = image_hw
        feat_h, feat_w = feature_hw

        if self.pixel_mode == "official":
            # 与官方 frustum 构造的 linspace 方式对齐
            u = torch.linspace(
                0, image_w - 1, feat_w,
                device=device, dtype=dtype,
            )
            v = torch.linspace(
                0, image_h - 1, feat_h,
                device=device, dtype=dtype,
            )

        elif self.pixel_mode == "cell_center":
            # 特征网格中心采样
            stride_w = image_w / feat_w
            stride_h = image_h / feat_h

            u = (
                (torch.arange(
                    feat_w, device=device, dtype=dtype
                ) + 0.5) * stride_w - 0.5
            )
            v = (
                (torch.arange(
                    feat_h, device=device, dtype=dtype
                ) + 0.5) * stride_h - 0.5
            )

        else:
            raise ValueError("Unknown pixel_mode")

        vv, uu = torch.meshgrid(
            v, u, indexing="ij"
        )

        return uu, vv

    def forward(
        self,
        K,
        world2cam,
        image_hw,
        feature_hw,
        ground_z=-30.0,
        ida=None,
    ):
        B = K.shape[0]
        Hf, Wf = feature_hw
        D = self.height_samples.numel()

        assert K.shape == (B, 3, 3)
        assert world2cam.shape == (B, 4, 4)

        if torch.any(K[:, 0, 1].abs() > 1e-6):
            raise ValueError(
                "Current projection expects zero K skew"
            )

        uu, vv = self.create_pixel_grid(
            image_hw, feature_hw,
            K.device, K.dtype,
        )

        # 原始特征网格对应的像素坐标
        pixel_u = uu.unsqueeze(0).expand(B, -1, -1)
        pixel_v = vv.unsqueeze(0).expand(B, -1, -1)

        # 撤销可选的图像增强变换
        ray_u, ray_v = pixel_u, pixel_v

        if ida is not None:
            assert ida.shape == (B, 4, 4)

            zero = torch.zeros_like(pixel_u)
            one = torch.ones_like(pixel_u)

            pixel_h = torch.stack(
                [pixel_u, pixel_v, zero, one],
                dim=-1,
            )

            inv_ida = torch.linalg.inv(ida)

            unaugmented = torch.einsum(
                "bij,bhwj->bhwi",
                inv_ida,
                pixel_h,
            )

            ray_u = (
                unaugmented[..., 0]
                / unaugmented[..., 3]
            )
            ray_v = (
                unaugmented[..., 1]
                / unaugmented[..., 3]
            )

        fx = K[:, 0, 0, None, None]
        fy = K[:, 1, 1, None, None]
        cx = K[:, 0, 2, None, None]
        cy = K[:, 1, 2, None, None]

        # 特别注意负号：
        # 相机 Y 左，所以像素 u 向右时 Y 减小
        directions_cam = torch.stack(
            [
                torch.ones_like(ray_u),
                (cx - ray_u) / fx,
                (cy - ray_v) / fy,
            ],
            dim=-1,
        )

        # 相机 -> 世界
        cam2world = torch.linalg.inv(world2cam)

        rotation = cam2world[:, :3, :3]
        origin = cam2world[:, :3, 3]

        directions_world = torch.einsum(
            "bij,bhwj->bhwi",
            rotation,
            directions_cam,
        )

        # 每个高度分箱对应一个世界 Z 平面
        height_samples = self.height_samples.to(
            device=K.device,
            dtype=K.dtype,
        )

        plane_z = (
            ground_z
            + height_samples.view(1, D, 1, 1)
        )

        origin_z = origin[:, 2].view(B, 1, 1, 1)

        direction_z = (
            directions_world[..., 2]
            .unsqueeze(1)
        )

        # 避免水平射线除零
        good_direction = direction_z.abs() > 1e-6

        safe_direction = torch.where(
            good_direction,
            direction_z,
            torch.ones_like(direction_z),
        )

        distance = (
            plane_z - origin_z
        ) / safe_direction

        # 仅保留相机前方、在最大计算距离内的交点
        valid = (
            good_direction
            & (distance > 1e-4)
            & (distance < self.max_forward)
            & torch.isfinite(distance)
        )

        xyz = (
            origin[:, None, None, None, :]
            + distance[..., None]
            * directions_world[:, None, :, :, :]
        )

        valid = valid & torch.isfinite(xyz).all(
            dim=-1
        )

        xyz = torch.where(
            valid[..., None],
            xyz,
            torch.zeros_like(xyz),
        )

        assert xyz.shape == (B, D, Hf, Wf, 3)
        assert valid.shape == (B, D, Hf, Wf)

        return {
            "xyz": xyz,
            "valid": valid,
            "heights": height_samples,
            "pixel_u": pixel_u,
            "pixel_v": pixel_v,
            "ray_u": ray_u,
            "ray_v": ray_v,
        }
