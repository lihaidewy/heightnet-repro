
import torch


class BEVTargetEncoder:
    """将世界坐标系中的车辆3D Box转换为BEV监督标签。"""

    def __init__(
        self,
        x_bound=(40.0, 168.0),
        y_bound=(-32.0, 48.0),
        resolution=1.0,
        sigma=1.0,
    ):
        self.x_min, self.x_max = x_bound
        self.y_min, self.y_max = y_bound
        self.resolution = resolution
        self.sigma = sigma

        self.nx = round(
            (self.x_max - self.x_min) / resolution
        )
        self.ny = round(
            (self.y_max - self.y_min) / resolution
        )

    def __call__(self, boxes):
        """
        boxes: [N,7]
            x, y, z, length, width, height, yaw

        返回:
            heatmap:   [1, Ny, Nx]
            indices:   [M]
            offsets:   [M, 2]
            log_sizes: [M, 3]
            yaws:      [M, 2] (sin, cos)

        M为落在BEV范围内的车辆数量。
        """
        boxes = torch.as_tensor(
            boxes, dtype=torch.float32
        ).reshape(-1, 7)

        heatmap = torch.zeros(
            1, self.ny, self.nx
        )

        indices = []
        offsets = []
        log_sizes = []
        yaws = []

        for box in boxes:
            x, y, z, length, width, height, yaw = box

            if not torch.isfinite(box).all():
                continue

            if (box[3:6] <= 0).any():
                continue

            gx = (x.item() - self.x_min) / self.resolution
            gy = (y.item() - self.y_min) / self.resolution

            ix = int(gx // 1)
            iy = int(gy // 1)

            if not (0 <= ix < self.nx and 0 <= iy < self.ny):
                continue

            # 1. 车辆中心的扁平化BEV索引
            indices.append(iy * self.nx + ix)

            # 2. 中心在网格单元内的亚网格偏移量
            offsets.append([
                gx - ix,
                gy - iy
            ])

            # 3. 尺寸使用对数表示
            log_sizes.append(
                torch.log(box[3:6]).tolist()
            )

            # 4. 航向角表示
            yaws.append([
                torch.sin(yaw).item(),
                torch.cos(yaw).item()
            ])

            # 5. 在车辆中心附近绘制高斯热力图
            radius = 3
            x0 = max(0, ix - radius)
            x1 = min(self.nx, ix + radius + 1)
            y0 = max(0, iy - radius)
            y1 = min(self.ny, iy + radius + 1)

            yy, xx = torch.meshgrid(
                torch.arange(y0, y1),
                torch.arange(x0, x1),
                indexing="ij"
            )

            gaussian = torch.exp(
                -(
                    (xx - ix).float().square()
                    + (yy - iy).float().square()
                ) / (2 * self.sigma**2)
            )

            patch = heatmap[0, y0:y1, x0:x1]
            patch.copy_(torch.maximum(patch, gaussian))

        return {
            "heatmap": heatmap,
            "indices": torch.tensor(
                indices, dtype=torch.long
            ),
            "offsets": torch.tensor(
                offsets, dtype=torch.float32
            ).reshape(-1, 2),
            "log_sizes": torch.tensor(
                log_sizes, dtype=torch.float32
            ).reshape(-1, 3),
            "yaws": torch.tensor(
                yaws, dtype=torch.float32
            ).reshape(-1, 2),
        }
