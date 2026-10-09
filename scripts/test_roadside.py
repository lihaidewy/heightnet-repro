
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.roadside import RoadsideHeightNet


def main():
    device = "cuda"

    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(288, 512)
    )

    # 使用已经验证过几何投影的第257帧
    index = next(
        i for i, (fid, _, _) in enumerate(dataset.samples)
        if fid == "257"
    )

    sample = dataset[index]

    image = sample["image"].unsqueeze(0).to(device)
    K = sample["intrinsics"].unsqueeze(0).to(device)
    T = sample["extrinsics_raw"].unsqueeze(0).to(device)

    model = RoadsideHeightNet().to(device)
    model.train()

    bev, debug = model(
        images=image,
        K=K,
        T=T,
        ground_z=-30.0,
        return_debug=True,
    )

    print("帧号:", sample["frame_id"])
    print("RGB:", image.shape)
    print("Context:", debug["context"].shape)
    print("Height:", debug["height_prob"].shape)
    print("XYZ:", debug["xyz"].shape)
    print("BEV:", bev.shape)

    # 验证高度概率
    prob_sum = debug["height_prob"].sum(dim=1)

    assert torch.allclose(
        prob_sum,
        torch.ones_like(prob_sum),
        atol=1e-5
    )

    assert bev.shape == (1, 80, 160, 160)
    assert torch.isfinite(bev).all()

    # 仅用来验证计算图，不是实际训练损失
    loss = bev.square().mean()
    loss.backward()

    backbone_grad = model.backbone.stem[0].weight.grad
    height_grad = model.heightnet.height_head.weight.grad
    aspp_grad = model.heightnet.height_aspp.project[0].weight.grad

    for name, grad in [
        ("Backbone", backbone_grad),
        ("Height Head", height_grad),
        ("ASPP", aspp_grad),
    ]:
        ok = (
            grad is not None
            and torch.isfinite(grad).all().item()
            and grad.abs().sum().item() > 0
        )
        print(f"{name} 有效梯度:", ok)
        assert ok, f"{name} 梯度检查失败"

    if torch.cuda.is_available():
        print(
            "GPU峰值显存(MB):",
            round(
                torch.cuda.max_memory_allocated() / 1024**2,
                2
            )
        )

    print("RoadsideHeightNet 整合测试通过")


if __name__ == "__main__":
    main()
