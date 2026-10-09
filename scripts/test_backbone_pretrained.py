
import torch

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.backbone import ImageBackboneFPN


def main():
    device = "cuda"

    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    sample = dataset[0]

    image = sample["image"].unsqueeze(0).to(device)

    print("帧号:", sample["frame_id"])
    print("RGB:", image.shape)
    print(
        "RGB数值范围:",
        image.min().item(),
        image.max().item(),
    )

    assert image.min() >= 0.0
    assert image.max() <= 1.0

    model = ImageBackboneFPN(
        output_channels=512,
        pretrained=True,
        depth=50,
    ).to(device)

    model.train()

    # 观察各级 ResNet 特征
    stage_shapes = {}

    handles = []

    for name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, name)

        def make_hook(stage_name):
            def hook(module, inputs, output):
                stage_shapes[stage_name] = tuple(
                    output.shape
                )
            return hook

        handles.append(
            layer.register_forward_hook(
                make_hook(name)
            )
        )

    features = model(image)

    for handle in handles:
        handle.remove()

    print("\nResNet50特征:")
    for name, shape in stage_shapes.items():
        print(name, shape)

    print("\nSECOND-FPN输出:", features.shape)

    assert stage_shapes["layer1"] == (
        1, 256, 144, 256
    )
    assert stage_shapes["layer2"] == (
        1, 512, 72, 128
    )
    assert stage_shapes["layer3"] == (
        1, 1024, 36, 64
    )
    assert stage_shapes["layer4"] == (
        1, 2048, 18, 32
    )

    assert features.shape == (
        1, 512, 36, 64
    )

    # 梯度测试
    loss = features.square().mean()
    loss.backward()

    backbone_grad = (
        model.stem[0].weight.grad
    )

    fpn_grad = (
        model.fpn.deblocks[0][0].weight.grad
    )

    print(
        "ResNet50梯度有效:",
        backbone_grad is not None
        and torch.isfinite(backbone_grad).all().item()
    )

    print(
        "SECOND-FPN梯度有效:",
        fpn_grad is not None
        and torch.isfinite(fpn_grad).all().item()
    )

    assert backbone_grad is not None
    assert fpn_grad is not None
    assert torch.isfinite(backbone_grad).all()
    assert torch.isfinite(fpn_grad).all()

    print(
        "\nGPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated() / 1024**2,
            2,
        )
    )

    print("\nResNet50 + SECOND-FPN 测试通过")


if __name__ == "__main__":
    main()
