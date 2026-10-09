
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.backbone import ImageBackboneFPN
from src.heightnet.model import HeightNet, lift_features


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取真实数据
    dataset = HeightNetDataset("data/S3S")
    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)
    K = sample["intrinsics"].unsqueeze(0).to(device)
    T = sample["extrinsics_raw"].unsqueeze(0).to(device)

    print("帧号:", sample["frame_id"])
    print("原始输入:", image.shape)

    # Backbone + FPN
    backbone = ImageBackboneFPN(
        output_channels=512,
        pretrained=False
    ).to(device)

    # HeightNet
    heightnet = HeightNet(
        in_channels=512,
        mid_channels=128,
        context_channels=80,
        num_height_bins=90
    ).to(device)

    backbone.train()
    heightnet.train()

    # 真实图像特征
    features = backbone(image)

    print("Backbone特征:", features.shape)

    # HeightNet 前向计算
    context, height_prob, logits = heightnet(
        features,
        K,
        T,
        image_size=image.shape[-2:]
    )

    print("Context:", context.shape)
    print("Height probability:", height_prob.shape)

    # 外积形状检查
    with torch.no_grad():
        lifted = lift_features(
            context.detach(),
            height_prob.detach()
        )
        print("Lifted features:", lifted.shape)

    # 测试梯度是否能传回 Backbone
    # 这里只是计算图测试，不是实际训练损失
    loss = (
        context.square().mean()
        + logits.square().mean()
    )

    loss.backward()

    print(
        "Backbone梯度:",
        backbone.stem[0].weight.grad is not None
    )

    print(
        "HeightNet梯度:",
        heightnet.height_head.weight.grad is not None
    )

    print(
        "ASPP梯度:",
        heightnet.height_aspp.project[0].weight.grad
        is not None
    )

    print("真实图像前向与反向计算完成")


if __name__ == "__main__":
    main()
