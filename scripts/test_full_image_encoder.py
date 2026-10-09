
import torch
import torch.nn.functional as F

from torchvision.models import (
    resnet50,
    ResNet50_Weights,
)

from src.heightnet.dataset import HeightNetDataset
from src.heightnet.full_image_encoder import (
    FullImageHeightEncoder,
)
from src.heightnet.heightnet_full import (
    build_camera_condition,
)


def main():
    torch.manual_seed(42)

    device = torch.device("cuda")

    dataset = HeightNetDataset(
        "data/S3S",
        image_size=(576, 1024),
    )

    samples = [dataset[0], dataset[1]]

    images = torch.stack(
        [s["image"] for s in samples]
    ).to(device)

    K = torch.stack([
        torch.as_tensor(
            s["intrinsics"], dtype=torch.float32
        )
        for s in samples
    ]).to(device)

    world2cam = torch.stack([
        torch.as_tensor(
            s["extrinsics_raw"], dtype=torch.float32
        )
        for s in samples
    ]).to(device)

    print("RGB:", tuple(images.shape))
    print("K:", tuple(K.shape))
    print("world2cam:", tuple(world2cam.shape))

    assert images.shape == (2, 3, 576, 1024)
    assert K.shape == (2, 3, 3)
    assert world2cam.shape == (2, 4, 4)

    # 检查 camera -> world 的变换
    condition = build_camera_condition(
        K, world2cam
    )

    extracted = condition[:, 15:].reshape(2, 3, 4)
    expected = torch.linalg.inv(
        world2cam
    )[:, :3, :4]

    assert torch.allclose(
        extracted, expected, atol=1e-5
    )
    print("27维相机外参编码检查通过")

    model = FullImageHeightEncoder(
        backbone_depth=50,
        backbone_pretrained=True,
    ).to(device)

    # 检查确实加载了 ImageNet 预训练权重
    reference = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V1
    )

    loaded = torch.equal(
        model.backbone.stem[0].weight.detach().cpu(),
        reference.conv1.weight.detach().cpu(),
    )

    assert loaded
    del reference

    print("ImageNet预训练权重检查通过")

    # 测试1：正式分辨率推理
    model.eval()

    with torch.inference_mode():
        predictions = model(
            images[:1],
            K[:1],
            world2cam[:1],
        )

        for name, value in predictions.items():
            print(name, tuple(value.shape))
            assert torch.isfinite(value).all()

        assert predictions["height_prob"].shape == (
            1, 90, 36, 64
        )
        assert predictions["context"].shape == (
            1, 80, 36, 64
        )

        prob_sum = predictions[
            "height_prob"
        ].sum(dim=1)

        assert torch.allclose(
            prob_sum,
            torch.ones_like(prob_sum),
            atol=1e-5,
        )

    del predictions

    print("576x1024真实图像推理通过")

    # 测试2：降低分辨率，检查完整反向传播
    # 缩放图像时必须同步缩放内参
    images_small = F.interpolate(
        images,
        size=(288, 512),
        mode="bilinear",
        align_corners=False,
    )

    K_small = K.clone()
    K_small[:, 0, :] *= 0.5
    K_small[:, 1, :] *= 0.5

    model.train()
    model.zero_grad(set_to_none=True)

    predictions = model(
        images_small,
        K_small,
        world2cam,
    )

    loss = (
        predictions["height_logits"].square().mean()
        + predictions["context"].square().mean()
    )

    loss.backward()

    checks = {
        "ResNet50": model.backbone.stem[0].weight,
        "SECOND-FPN":
            model.backbone.fpn.deblocks[0][0].weight,
        "HeightNet":
            model.heightnet.reduce_conv[0].weight,
        "Context Branch":
            model.heightnet.context_conv.weight,
        "Height Branch":
            model.heightnet.height_layer.weight,
        "DCN":
            model.heightnet.height_conv[-1].weight,
    }

    for name, parameter in checks.items():
        ok = (
            parameter.grad is not None
            and torch.isfinite(
                parameter.grad
            ).all().item()
        )

        print(f"{name}梯度有效: {ok}")
        assert ok

    print(
        "GPU峰值分配显存(MB):",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**2,
            2,
        ),
    )

    print(
        "完整图像编码器前向、反向测试通过"
    )


if __name__ == "__main__":
    main()
