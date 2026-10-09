import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import SplitHeightNetDataset
from src.heightnet.full_detector import FullBEVHeightDetector


# ============================================================
# 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="outputs/train_heightnet_full/best.pth"
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
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--match-distance",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--nms-radius",
        type=float,
        default=1.5
    )

    return parser.parse_args()


# ============================================================
# DataLoader
# ============================================================

def collate_fn(batch):

    return {

        "image":
            torch.stack(
                [
                    x["image"]
                    for x in batch
                ]
            ),

        "intrinsics":
            torch.stack(
                [
                    x["intrinsics"]
                    for x in batch
                ]
            ),

        "extrinsics_raw":
            torch.stack(
                [
                    x["extrinsics_raw"]
                    for x in batch
                ]
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
            ]
    }


# ============================================================
# Heatmap NMS
# ============================================================

def local_peak_nms(
    heatmap,
    kernel=3
):

    pad = kernel // 2

    pooled = F.max_pool2d(
        heatmap,
        kernel_size=kernel,
        stride=1,
        padding=pad
    )

    keep = (
        pooled == heatmap
    )

    return heatmap * keep



# ============================================================
# Circle NMS
# ============================================================

def circle_nms(
    candidates,
    radius=1.5
):

    if len(candidates) == 0:
        return []


    candidates = sorted(
        candidates,
        key=lambda x:x["score"],
        reverse=True
    )


    keep=[]


    for item in candidates:

        valid=True

        for selected in keep:

            d=torch.norm(
                item["xyz"][:2]
                -
                selected["xyz"][:2]
            )

            if d < radius:

                valid=False
                break


        if valid:

            keep.append(item)


    return keep



# ============================================================
# CenterPoint Decode
# ============================================================

def decode_predictions(
    prediction,
    score_threshold,
    nms_radius
):

    heatmap = (
        prediction["heatmap"]
        .sigmoid()
    )

    heatmap = local_peak_nms(
        heatmap
    )


    B = heatmap.shape[0]

    results=[]


    for b in range(B):

        hm=heatmap[b,0]


        scores, inds=torch.topk(
            hm.reshape(-1),
            k=100
        )


        candidates=[]


        for score, ind in zip(
            scores,
            inds
        ):

            score=float(score)

            if score < score_threshold:
                continue


            iy = (
                ind // hm.shape[1]
            )

            ix = (
                ind % hm.shape[1]
            )


            dx = prediction["reg"][
                b,0,iy,ix
            ]

            dy = prediction["reg"][
                b,1,iy,ix
            ]


            # BEV坐标
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
                prediction["height"]
                [b,0,iy,ix]
                -30.0
            )


            size=torch.exp(
                prediction["dim"]
                [b,:,iy,ix]
            )


            yaw=torch.atan2(
                prediction["rot"]
                [b,0,iy,ix],
                prediction["rot"]
                [b,1,iy,ix]
            )


            candidates.append(
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
                    score,

                    "size":
                    size.cpu(),

                    "yaw":
                    yaw.item()
                }
            )


        results.append(
            circle_nms(
                candidates,
                radius=nms_radius
            )
        )


    return results



# ============================================================
# 匹配
# ============================================================

def match_predictions(
    predictions,
    gt_boxes,
    threshold
):

    num_gt=len(gt_boxes)
    num_pred=len(predictions)


    if num_gt==0:

        return [], num_pred, 0


    if num_pred==0:

        return [],0,num_gt


    pred=torch.stack(
        [
            p["xyz"][:2]
            for p in predictions
        ]
    )


    gt=gt_boxes[:,:2]


    cost=torch.cdist(
        pred,
        gt
    ).cpu().numpy()


    rows,cols=linear_sum_assignment(
        cost
    )


    matches=[]

    used_pred=set()
    used_gt=set()


    for r,c in zip(rows,cols):

        if cost[r,c] <= threshold:

            matches.append(
                (
                    r,
                    c,
                    cost[r,c]
                )
            )

            used_pred.add(r)
            used_gt.add(c)


    fp=num_pred-len(used_pred)

    fn=num_gt-len(used_gt)


    return matches,fp,fn



# ============================================================
# 相机距离
# ============================================================

def camera_distance(
    xyz_world,
    world2cam
):

    device = world2cam.device


    point = torch.cat(
        [
            xyz_world.to(device),
            torch.ones(
                1,
                device=device
            )
        ]
    )


    cam = world2cam @ point


    x = cam[0]
    z = cam[2]


    return torch.sqrt(
        x*x + z*z
    ).item()



# ============================================================
# 统计
# ============================================================

def print_error(
    name,
    errors
):

    if len(errors)==0:

        print(
            name,
            ": empty"
        )

        return


    errors=np.array(errors)


    print("\n"+name)

    print(
        "Count:",
        len(errors)
    )

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


    for t in [
        0.5,
        1.0,
        2.0
    ]:

        print(
            f"<= {t}m:",
            np.mean(errors<=t)
        )



# ============================================================
# 主程序
# ============================================================

def evaluate_threshold(
    model,
    loader,
    device,
    score_threshold,
    args
):

    total_tp=0
    total_fp=0
    total_fn=0


    tp_errors=[]

    all_gt_errors=[]


    distance_bins={

        "40-80m":[],
        "80-120m":[],
        "120m+":[]

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


            outputs=model(
                images,
                K,
                T
            )


            predictions=decode_predictions(
                outputs["predictions"],
                score_threshold,
                args.nms_radius
            )


            for i in range(
                len(predictions)
            ):

                gt=batch["boxes_raw"][i]


                matches,fp,fn=match_predictions(
                    predictions[i],
                    gt,
                    args.match_distance
                )


                total_tp+=len(matches)
                total_fp+=fp
                total_fn+=fn


                matched_gt=set()


                for p,g,d in matches:

                    tp_errors.append(d)

                    matched_gt.add(g)


                # All GT
                for gi in range(
                    len(gt)
                ):

                    if gi in matched_gt:

                        continue

                    all_gt_errors.append(
                        float("inf")
                    )


                for p,g,d in matches:

                    xyz=gt[g,:3]

                    dist=camera_distance(
                        xyz,
                        T[i]
                    )


                    if dist < 80:

                        distance_bins[
                            "40-80m"
                        ].append(d)

                    elif dist <120:

                        distance_bins[
                            "80-120m"
                        ].append(d)

                    else:

                        distance_bins[
                            "120m+"
                        ].append(d)


            if (step+1)%20==0:

                print(
                    f"Eval {step+1}/{len(loader)}"
                )


    precision=(
        total_tp/
        max(
            total_tp+total_fp,
            1
        )
    )


    recall=(
        total_tp/
        max(
            total_tp+total_fn,
            1
        )
    )


    f1=(
        2*precision*recall/
        max(
            precision+recall,
            1e-9
        )
    )


    print("\n==============================")
    print(
        "Score:",
        score_threshold
    )

    print(
        "TP:",
        total_tp,
        "FP:",
        total_fp,
        "FN:",
        total_fn
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


    print_error(
        "TP localization",
        tp_errors
    )


    print_error(
        "All-GT localization",
        all_gt_errors
    )


    print("\nDistance bins")

    for k,v in distance_bins.items():

        print_error(
            k,
            v
        )



# ============================================================
# main
# ============================================================

def main():

    args=parse_args()


    device=torch.device(
        "cuda"
    )


    dataset=SplitHeightNetDataset(
        args.val_split,
        image_size=(
            args.image_height,
            args.image_width
        )
    )


    loader=DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )


    model=FullBEVHeightDetector(
        backbone_pretrained=False,
        backbone_depth=50,
        ground_z=-30
    ).to(device)


    ckpt=torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )


    model.load_state_dict(
        ckpt["model"]
    )


    model.eval()


    print(
        "加载模型 Epoch:",
        ckpt["epoch"]
    )

    print(
        "Best val loss:",
        ckpt["best_val"]
    )


    for threshold in [
        0.1,
        0.2,
        0.3,
        0.5
    ]:

        evaluate_threshold(
            model,
            loader,
            device,
            threshold,
            args
        )



if __name__=="__main__":

    main()