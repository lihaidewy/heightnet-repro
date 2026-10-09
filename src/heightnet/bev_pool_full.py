
import math

import torch
import torch.nn as nn


class VoxelPoolingFull(nn.Module):
    """
    与 BEVHeight 官方体素池化数学机制对齐的
    PyTorch 实现。

    使用三维体素索引过滤有效点，
    沿 Z 方向直接累加到二维 BEV。

    输入:
        context:     [B, C, H, W]
        height_prob: [B, D, H, W]
        xyz:         [B, D, H, W, 3]
        valid:       [B, D, H, W]

    输出:
        BEV:         [B, C, Ny, Nx]

    使用 FP32 累加，支持反向传播。
    """

    def __init__(
        self,
        x_bound=(40.0, 168.0, 1.0),
        y_bound=(-32.0, 48.0, 1.0),
        z_bound=(-30.0, -26.0, 4.0),
        chunk_bins=12,
    ):
        super().__init__()

        bounds = [x_bound, y_bound, z_bound]
        counts = []

        for lo, hi, step in bounds:
            if hi <= lo or step <= 0:
                raise ValueError("Invalid voxel bounds")

            n = (hi - lo) / step

            if not math.isclose(
                n, round(n), rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError(
                    "Voxel range must be divisible by step"
                )

            counts.append(int(round(n)))

        self.nx, self.ny, self.nz = counts

        if chunk_bins < 1:
            raise ValueError("chunk_bins must be >= 1")

        self.chunk_bins = chunk_bins

        self.register_buffer(
            "xyz_min",
            torch.tensor(
                [x_bound[0], y_bound[0], z_bound[0]],
                dtype=torch.float32,
            ),
        )

        self.register_buffer(
            "voxel_step",
            torch.tensor(
                [x_bound[2], y_bound[2], z_bound[2]],
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        context,
        height_prob,
        xyz,
        valid,
    ):
        B, C, H, W = context.shape
        D = height_prob.shape[1]

        if height_prob.shape != (B, D, H, W):
            raise ValueError("Invalid height_prob shape")

        if xyz.shape != (B, D, H, W, 3):
            raise ValueError("Invalid xyz shape")

        if valid.shape != (B, D, H, W):
            raise ValueError("Invalid valid shape")

        devices = [
            context.device,
            height_prob.device,
            xyz.device,
            valid.device,
        ]

        if len(set(devices)) != 1:
            raise ValueError("Inputs must be on same device")

        if valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")

        # 直接创建二维 BEV 累加缓冲区。
        # 输出统一用 FP32，避免混合精度累加误差。
        output = torch.zeros(
            (B * self.ny * self.nx, C),
            device=context.device,
            dtype=torch.float32,
        )

        minimum = self.xyz_min.to(dtype=xyz.dtype)
        step = self.voxel_step.to(dtype=xyz.dtype)

        # 分高度块处理，避免一次性构造
        # [B, C, D, H, W] 的大张量。
        for d0 in range(0, D, self.chunk_bins):
            d1 = min(d0 + self.chunk_bins, D)
            d = d1 - d0

            points = xyz[:, d0:d1]

            # 体素索引不参与梯度传播；
            # 与官方 CUDA 算子的处理原则一致。
            with torch.no_grad():
                finite = torch.isfinite(points).all(
                    dim=-1
                )

                safe_points = torch.where(
                    finite[..., None],
                    points,
                    minimum,
                )

                voxel_xyz = torch.floor(
                    (safe_points - minimum) / step
                ).long()

                ix = voxel_xyz[..., 0]
                iy = voxel_xyz[..., 1]
                iz = voxel_xyz[..., 2]

                keep = (
                    valid[:, d0:d1]
                    & finite
                    & (ix >= 0)
                    & (ix < self.nx)
                    & (iy >= 0)
                    & (iy < self.ny)
                    & (iz >= 0)
                    & (iz < self.nz)
                )

                batch_index = torch.arange(
                    B, device=context.device
                ).view(B, 1, 1, 1).expand(
                    B, d, H, W
                )

                # 输出按 [B, Ny, Nx, C] 展平。
                # Z 仅参与有效性判断，随后直接聚合。
                flat_index = (
                    batch_index * (self.ny * self.nx)
                    + iy * self.nx
                    + ix
                ).reshape(-1)

                keep = keep.reshape(-1)

            # 高度概率 × 上下文特征
            # [B,C,1,H,W] * [B,1,d,H,W]
            lifted = (
                context.float().unsqueeze(2)
                * height_prob[:, d0:d1]
                    .float().unsqueeze(1)
            )

            # [B,d,H,W,C] -> [B*d*H*W,C]
            lifted = (
                lifted.permute(0, 2, 3, 4, 1)
                .reshape(-1, C)
            )

            # 累加具有可用梯度的 lifted features
            output.index_add_(
                0,
                flat_index[keep],
                lifted[keep],
            )

        bev = (
            output.view(B, self.ny, self.nx, C)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return bev
