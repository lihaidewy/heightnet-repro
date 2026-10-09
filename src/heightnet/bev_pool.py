
import torch
import torch.nn as nn


class BEVPool(nn.Module):
    """
    基础 BEV Pooling：将三维候选特征求和到 BEV 网格。

    features: [B, C, D, Hf, Wf]
    xyz:      [B, D, Hf, Wf, 3]
    valid:    [B, D, Hf, Wf]

    输出:
        bev:  [B, C, Ny, Nx]

    约定:
        世界 X -> BEV 列
        世界 Y -> BEV 行
    """

    def __init__(
        self,
        x_bound=(0.0, 160.0),
        y_bound=(-80.0, 80.0),
        resolution=1.0,
    ):
        super().__init__()

        self.x_min, self.x_max = x_bound
        self.y_min, self.y_max = y_bound
        self.resolution = resolution

        self.nx = round(
            (self.x_max - self.x_min) / resolution
        )
        self.ny = round(
            (self.y_max - self.y_min) / resolution
        )

        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Invalid BEV bounds")

    def forward(self, features, xyz, valid):
        B, C, D, Hf, Wf = features.shape

        assert xyz.shape == (B, D, Hf, Wf, 3)
        assert valid.shape == (B, D, Hf, Wf)

        x = xyz[..., 0]
        y = xyz[..., 1]

        ix = torch.floor(
            (x - self.x_min) / self.resolution
        ).long()

        iy = torch.floor(
            (y - self.y_min) / self.resolution
        ).long()

        # 只保留有限、有效、落入 BEV 范围的点
        mask = (
            valid
            & torch.isfinite(x)
            & torch.isfinite(y)
            & (ix >= 0)
            & (ix < self.nx)
            & (iy >= 0)
            & (iy < self.ny)
        )

        bev_outputs = []

        for b in range(B):
            # 将三维网格转为一维 BEV 索引
            flat_index = (
                iy[b] * self.nx + ix[b]
            ).reshape(-1)

            keep = mask[b].reshape(-1)

            # [C, D*Hf*Wf]
            feat_flat = features[b].reshape(C, -1)

            # 创建 BEV 特征
            bev_flat = features.new_zeros(
                C, self.ny * self.nx
            )

            if keep.any():
                selected_index = flat_index[keep]

                selected_feat = feat_flat[:, keep]

                bev_flat.scatter_add_(
                    dim=1,
                    index=selected_index.unsqueeze(0).expand(
                        C, -1
                    ),
                    src=selected_feat
                )

            bev_outputs.append(
                bev_flat.reshape(C, self.ny, self.nx)
            )

        return torch.stack(bev_outputs, dim=0)
