
import math
import torch


def gaussian_radius(height, width, overlap=0.1):
    """与 MMDetection 的 CenterNet 半径计算一致。"""
    h, w = float(height), float(width)

    b1 = h + w
    c1 = h * w * (1 - overlap) / (1 + overlap)
    r1 = (b1 - math.sqrt(max(0.0, b1*b1 - 4*c1))) / 2

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - overlap) * h * w
    r2 = (
        b2 - math.sqrt(max(0.0, b2*b2 - 4*a2*c2))
    ) / (2*a2)

    a3 = 4 * overlap
    b3 = -2 * overlap * (h + w)
    c3 = (overlap - 1) * h * w
    r3 = (
        -b3 + math.sqrt(max(0.0, b3*b3 - 4*a3*c3))
    ) / (2*a3)

    return min(r1, r2, r3)


@torch.no_grad()
def draw_gaussian(heatmap, ix, iy, radius):
    """在 heatmap[iy, ix] 附近绘制高斯峰。"""
    H, W = heatmap.shape

    left = min(ix, radius)
    right = min(W - ix - 1, radius)
    top = min(iy, radius)
    bottom = min(H - iy - 1, radius)

    xs = torch.arange(
        -left, right + 1, device=heatmap.device
    )
    ys = torch.arange(
        -top, bottom + 1, device=heatmap.device
    )

    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    sigma = (2 * radius + 1) / 6

    gaussian = torch.exp(
        -(xx.float()**2 + yy.float()**2)
        / (2 * sigma**2)
    )

    region = heatmap[
        iy-top:iy+bottom+1,
        ix-left:ix+right+1
    ]

    torch.maximum(region, gaussian, out=region)


class CenterTargetGenerator:
    """
    输入：每帧 [N,7] 的车辆框列表。
    格式：[x,y,z_bottom,length,width,height,yaw]

    输出：
        heatmap [B,1,Ny,Nx]
        anno    [B,max_objs,8]
        inds    [B,max_objs]
        mask    [B,max_objs]
    """

    def __init__(
        self,
        x_bound=(40.0, 168.0, 1.0),
        y_bound=(-32.0, 48.0, 1.0),
        ground_z=-30.0,
        max_objs=500,
        gaussian_overlap=0.1,
        min_radius=2,
    ):
        self.x_min, self.x_max, self.dx = x_bound
        self.y_min, self.y_max, self.dy = y_bound
        self.ground_z = ground_z
        self.max_objs = max_objs
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius

        self.nx = round(
            (self.x_max - self.x_min) / self.dx
        )
        self.ny = round(
            (self.y_max - self.y_min) / self.dy
        )

    @torch.no_grad()
    def __call__(self, gt_boxes_list, device):
        B = len(gt_boxes_list)

        heatmap = torch.zeros(
            B, 1, self.ny, self.nx,
            device=device, dtype=torch.float32
        )

        anno = torch.zeros(
            B, self.max_objs, 8,
            device=device, dtype=torch.float32
        )

        inds = torch.zeros(
            B, self.max_objs,
            device=device, dtype=torch.long
        )

        mask = torch.zeros(
            B, self.max_objs,
            device=device, dtype=torch.bool
        )

        for b, boxes in enumerate(gt_boxes_list):
            boxes = torch.as_tensor(
                boxes, dtype=torch.float32,
                device=device
            ).reshape(-1, 7)

            occupied = set()
            count = 0

            for box in boxes:
                if not torch.isfinite(box).all():
                    raise ValueError("Non-finite GT box")

                x, y, z, length, width, height, yaw = (
                    box.tolist()
                )

                if min(length, width, height) <= 0:
                    raise ValueError("Invalid GT box size")

                gx = (x - self.x_min) / self.dx
                gy = (y - self.y_min) / self.dy

                if not (
                    0 <= gx < self.nx
                    and 0 <= gy < self.ny
                ):
                    continue

                ix = math.floor(gx)
                iy = math.floor(gy)

                # 单个中心网格不能独立回归两辆车。
                if (iy, ix) in occupied:
                    raise RuntimeError(
                        f"Two GT centers share BEV cell "
                        f"({iy}, {ix}) in batch item {b}"
                    )

                occupied.add((iy, ix))

                if count >= self.max_objs:
                    raise RuntimeError(
                        "Too many GT objects"
                    )

                radius = gaussian_radius(
                    length / self.dx,
                    width / self.dy,
                    self.gaussian_overlap,
                )
                radius = max(
                    self.min_radius, int(radius)
                )

                draw_gaussian(
                    heatmap[b, 0],
                    ix, iy, radius
                )

                inds[b, count] = iy * self.nx + ix
                mask[b, count] = True

                anno[b, count] = torch.tensor(
                    [
                        gx - ix,
                        gy - iy,
                        z - self.ground_z,
                        math.log(length),
                        math.log(width),
                        math.log(height),
                        math.sin(yaw),
                        math.cos(yaw),
                    ],
                    device=device,
                    dtype=torch.float32,
                )

                count += 1

        return {
            "heatmap": heatmap,
            "anno": anno,
            "inds": inds,
            "mask": mask,
        }
