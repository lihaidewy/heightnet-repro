
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.geometry import build_height_frustum
from src.heightnet.height_bins import LIDHeightBins
from src.heightnet.bev_pool import BEVPool


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = HeightNetDataset("data/S3S")

    index = next(
        i for i, (frame_id, _, _) in enumerate(dataset.samples)
        if frame_id == "257"
    )
    sample = dataset[index]

    K = sample["intrinsics"].unsqueeze(0).to(device)
    T = sample["extrinsics_raw"].unsqueeze(0).to(device)

    lid = LIDHeightBins(
        num_bins=90,
        h_min=0.0,
        h_max=4.0
    ).to(device)

    # 生成三维坐标网格
    xyz, valid = build_height_frustum(
        K=K,
        T=T,
        height_centers=lid.centers,
        feature_hw=(36, 64),
        image_hw=(576, 1024),
        ground_z=-30.0
    )

    # 先用少量通道验证，避免不必要的显存占用
    context = torch.randn(
        1, 8, 36, 64,
        device=device,
        requires_grad=True
    )

    logits = torch.randn(
        1, 90, 36, 64,
        device=device,
        requires_grad=True
    )

    height_prob = logits.softmax(dim=1)

    features_3d = (
        context.unsqueeze(2)
        * height_prob.unsqueeze(1)
    )

    pool = BEVPool(
        x_bound=(0.0, 160.0),
        y_bound=(-80.0, 80.0),
        resolution=1.0
    ).to(device)

    bev = pool(features_3d, xyz, valid)

    print("三维特征:", features_3d.shape)
    print("三维坐标:", xyz.shape)
    print("BEV特征:", bev.shape)

    # 计算非空 BEV 单元数量
    occupied = (
        bev.abs().sum(dim=1) > 0
    ).sum().item()

    print("非零BEV单元数:", occupied)

    assert bev.shape == (1, 8, 160, 160)
    assert torch.isfinite(bev).all()
    assert occupied > 0

    # 测试梯度能否回传
    loss = bev.square().mean()
    loss.backward()

    print("Context梯度:", context.grad is not None)
    print("Height logits梯度:", logits.grad is not None)

    print("BEV Pooling 测试通过")


if __name__ == "__main__":
    main()
