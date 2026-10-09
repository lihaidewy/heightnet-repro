import argparse
import math
import torch
import numpy as np

from torch.utils.data import DataLoader

from src.heightnet.split_dataset import (
    SplitHeightNetDataset
)
from src.heightnet.full_detector import (
    FullBEVHeightDetector
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=
        "outputs/train_heightnet_full/best.pth"
    )

    parser.add_argument(
        "--val-split",
        default="data/splits/val.json"
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=576
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=1024
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--match-distance",
        type=float,
        default=2.0
    )

    return parser.parse_args()


def collate_fn(batch):

    return {
        "image":
            torch.stack(
                [x["image"] for x in batch]
            ),

        "intrinsics":
            torch.stack(
                [x["intrinsics"] for x in batch]
            ),

        "extrinsics_raw":
            torch.stack(
                [x["extrinsics_raw"] for x in batch]
            ),

        "boxes_raw":
            [
                x["boxes_raw"]
                for x in batch
            ],

        "frame_id":
            [
                x["frame_id"]
                for x in batch
            ],
    }


def decode_predictions(
    pred,
    score_threshold=0.1,
):

    heatmap = (
        pred["heatmap"]
        .sigmoid()
        .detach()
    )

    reg = (
        pred["reg"]
        .detach()
    )

    height = (
        pred["height"]
        .detach()
    )

    dim = (
        pred["dim"]
        .detach()
    )

    rot = (
        pred["rot"]
        .detach()
    )


    B = heatmap.shape[0]


    results=[]


    for b in range(B):

        hm = heatmap[b,0]


        scores, inds = torch.topk(
            hm.flatten(),
            k=min(
                50,
                hm.numel()
            )
        )


        vehicles=[]


        for score, ind in zip(
            scores,
            inds
        ):

            if score.item() < score_threshold:
                continue


            iy = (
                ind // hm.shape[1]
            )

            ix = (
                ind % hm.shape[1]
            )


            dx = reg[
                b,0,iy,ix
            ]

            dy = reg[
                b,1,iy,ix
            ]


            # BEV coordinate
            x = (
                40.0
                +
                ix.float()
                +
                dx
            )

            y = (
                -32.0
                +
                iy.float()
                +
                dy
            )


            z = (
                height[
                    b,0,iy,ix
                ]
                -30.0
            )


            l,w,h = torch.exp(
                dim[
                    b,:,iy,ix
                ]
            )


            sin_yaw = rot[
                b,0,iy,ix
            ]

            cos_yaw = rot[
                b,1,iy,ix
            ]

            yaw = torch.atan2(
                sin_yaw,
                cos_yaw
            )


            vehicles.append(
                {
                    "xyz":
                    torch.tensor(
                        [
                            x.item(),
                            y.item(),
                            z.item()
                        ]
                    ),

                    "score":
                    score.item(),

                    "size":
                    torch.tensor(
                        [
                            l.item(),
                            w.item(),
                            h.item()
                        ]
                    ),

                    "yaw":
                    yaw.item()
                }
            )


        results.append(
            vehicles
        )


    return results



def match_predictions(
    predictions,
    gt_boxes,
    threshold=2.0,
):

    if len(predictions)==0:
        return 0,0,len(gt_boxes),[]

    if len(gt_boxes)==0:
        return 0,len(predictions),0,[]


    pred_xyz = torch.stack(
        [
            x["xyz"]
            for x in predictions
        ]
    )


    gt_xyz = gt_boxes[:,:3]


    dist = torch.cdist(
        pred_xyz[:,:2],
        gt_xyz[:,:2]
    )


    used=set()

    errors=[]

    tp=0


    order = sorted(
        range(len(predictions)),
        key=lambda i:
        predictions[i]["score"],
        reverse=True
    )


    for i in order:

        values = dist[i]


        min_dist, idx = (
            values.min(0)
        )

        idx=int(idx)

        if (
            min_dist.item()
            <= threshold
            and idx not in used
        ):

            used.add(idx)

            tp+=1

            errors.append(
                min_dist.item()
            )


    fp=len(predictions)-tp

    fn=len(gt_boxes)-tp


    return (
        tp,
        fp,
        fn,
        errors
    )



def main():

    args=parse_args()

    device=torch.device("cuda")


    dataset=SplitHeightNetDataset(
        args.val_split,
        image_size=(
            args.image_height,
            args.image_width
        )
    )


    loader=DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )


    model=FullBEVHeightDetector(
        backbone_pretrained=False,
        backbone_depth=50,
        ground_z=-30
    ).to(device)


    checkpoint=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )


    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()


    print(
        "加载模型 Epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best val loss:",
        checkpoint["best_val"]
    )


    total_tp=0
    total_fp=0
    total_fn=0

    all_errors=[]

    all_gt_errors=[]


    distance_bins={
        "40-80m":[],
        "80-120m":[],
        "120m+":[],
    }


    with torch.no_grad():

        for step,batch in enumerate(loader):

            images=batch["image"].to(
                device
            )

            K=batch["intrinsics"].to(
                device
            )

            T=batch["extrinsics_raw"].to(
                device
            )


            # 输入尺寸如果不是训练尺寸
            # 这里保持576x1024

            outputs=model(
                images,
                K,
                T
            )


            predictions=decode_predictions(
                outputs["predictions"],
                args.score_threshold
            )


            for i in range(
                len(predictions)
            ):

                gt=batch["boxes_raw"][i]


                tp,fp,fn,errors=(
                    match_predictions(
                        predictions[i],
                        gt,
                        args.match_distance
                    )
                )


                total_tp+=tp
                total_fp+=fp
                total_fn+=fn


                all_errors.extend(
                    errors
                )


                # 所有GT最近预测误差
                if len(predictions[i]):

                    pred_xyz=torch.stack(
                        [
                            p["xyz"]
                            for p in predictions[i]
                        ]
                    )


                    gt_xyz=gt[:,:3]


                    d=torch.cdist(
                        gt_xyz[:,:2],
                        pred_xyz[:,:2]
                    )


                    min_d,_=d.min(1)


                    all_gt_errors.extend(
                        min_d.tolist()
                    )


                    for obj,e in zip(
                        gt,
                        min_d
                    ):

                        dist=torch.norm(
                            obj[:2]
                        ).item()


                        if dist<80:
                            distance_bins[
                                "40-80m"
                            ].append(e.item())

                        elif dist<120:
                            distance_bins[
                                "80-120m"
                            ].append(e.item())

                        else:
                            distance_bins[
                                "120m+"
                            ].append(e.item())


            if (step+1)%20==0:
                print(
                    f"Eval {step+1}/{len(loader)}"
                )


    precision=(
        total_tp/
        max(total_tp+total_fp,1)
    )

    recall=(
        total_tp/
        max(total_tp+total_fn,1)
    )


    f1=(
        2*precision*recall/
        max(precision+recall,1e-9)
    )


    print("\n========== Detection ==========")

    print(
        f"TP={total_tp} "
        f"FP={total_fp} "
        f"FN={total_fn}"
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1:",
        f1
    )


    def print_error(
        name,
        errors
    ):

        errors=np.array(errors)

        print("\n",name)

        print(
            "Mean:",
            errors.mean()
        )

        print(
            "Median:",
            np.median(errors)
        )

        print(
            "P90:",
            np.percentile(
                errors,
                90
            )
        )

        for th in [
            0.5,
            1.0,
            2.0
        ]:

            print(
                f"<= {th}m:",
                np.mean(
                    errors<=th
                )
            )


    print_error(
        "TP localization",
        all_errors
    )


    print_error(
        "All GT localization",
        all_gt_errors
    )


    print("\n========== Distance ==========")

    for k,v in distance_bins.items():

        if len(v):
            print_error(
                k,
                v
            )


if __name__=="__main__":
    main()