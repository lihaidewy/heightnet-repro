import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from src.heightnet.split_dataset import SplitHeightNetDataset
from src.heightnet.full_detector import FullBEVHeightDetector



# ==========================================================
# Args
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--checkpoint",
        default=
        "outputs/train_heightnet_full/best.pth"
    )


    parser.add_argument(
        "--val-split",
        default=
        "data/splits/val.json"
    )


    parser.add_argument(
        "--scene",
        default="S3S"
    )


    parser.add_argument(
        "--frame-id",
        default="257"
    )


    parser.add_argument(
        "--output-dir",
        default=
        "outputs/visualization"
    )


    parser.add_argument(
        "--score",
        type=float,
        default=0.2
    )


    parser.add_argument(
        "--match-distance",
        type=float,
        default=2.0
    )


    return parser.parse_args()



# ==========================================================
# Dataset
# ==========================================================

def collate_fn(batch):

    return batch[0]
import json


def find_sample(
    train_file,
    val_file,
    scene,
    frame_id
):


    files=[
        train_file,
        val_file
    ]


    for file in files:

        with open(
            file,
            "r"
        ) as f:

            records=json.load(f)


        for r in records:


            if (
                r["scene"]==scene
                and
                str(r["frame_id"])
                ==
                str(frame_id)
            ):

                return file


    return None


# ==========================================================
# Heatmap NMS
# ==========================================================

def local_peak_nms(
    heatmap,
    kernel=3
):

    pooled=F.max_pool2d(
        heatmap,
        kernel,
        stride=1,
        padding=1
    )


    keep = (
        pooled == heatmap
    )


    return heatmap*keep



# ==========================================================
# Circle NMS
# ==========================================================

def circle_nms(
    candidates,
    radius=1.5
):

    candidates=sorted(
        candidates,
        key=lambda x:x["score"],
        reverse=True
    )


    keep=[]


    for c in candidates:


        valid=True


        for k in keep:


            d=torch.norm(
                c["xyz"][:2]
                -
                k["xyz"][:2]
            )


            if d < radius:

                valid=False
                break


        if valid:

            keep.append(c)


    return keep



# ==========================================================
# Decode
# ==========================================================

def decode_predictions(
    pred,
    score_threshold=0.2
):

    heatmap=(
        pred["heatmap"]
        .sigmoid()
    )


    heatmap=local_peak_nms(
        heatmap
    )


    hm=heatmap[0,0]


    scores,inds=torch.topk(
        hm.reshape(-1),
        k=100
    )


    candidates=[]


    for score,ind in zip(
        scores,
        inds
    ):


        score=float(score)


        if score < score_threshold:
            continue



        iy=ind//hm.shape[1]

        ix=ind%hm.shape[1]



        x=(
            40
            +
            ix.float()
            +
            pred["reg"]
            [0,0,iy,ix]
        )


        y=(
            -32
            +
            iy.float()
            +
            pred["reg"]
            [0,1,iy,ix]
        )


        z=(
            pred["height"]
            [0,0,iy,ix]
            -
            30
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
            score

            }
        )


    return circle_nms(
        candidates
    )



# ==========================================================
# Match
# ==========================================================

def match_predictions(
    preds,
    gt,
    threshold=2.0
):

    pred_xyz=torch.stack(
        [
            p["xyz"][:2]
            for p in preds
        ]
    )


    gt_xyz=gt[:,:2]


    cost=torch.cdist(
        pred_xyz,
        gt_xyz
    ).numpy()


    rows,cols=linear_sum_assignment(
        cost
    )


    matches=[]


    for r,c in zip(rows,cols):


        if cost[r,c] <= threshold:


            matches.append(
                (
                    preds[r],
                    gt[c],
                    cost[r,c]
                )
            )


    return matches



# ==========================================================
# World -> Image projection
# ==========================================================

def project_points(
    xyz,
    K,
    world2cam
):

    device = world2cam.device


    xyz = xyz.to(device)


    pts = torch.cat(
        [
            xyz,
            torch.ones(
                1,
                device=device
            )
        ]
    )


    cam = world2cam @ pts


    if cam[2] <= 0:

        return None


    uv=K @ cam[:3]


    u=uv[0]/uv[2]

    v=uv[1]/uv[2]


    return (
        float(u),
        float(v)
    )



# ==========================================================
# RGB visualization
# ==========================================================

def draw_rgb(
    image,
    K,
    T,
    matches,
    save
):

    img=image.permute(
        1,2,0
    ).cpu().numpy()


    plt.figure(
        figsize=(12,7)
    )


    plt.imshow(
        img
    )



    for pred,gt,error in matches:


        gt_uv=project_points(
            gt[:3],
            K,
            T
        )


        pred_uv=project_points(
            pred["xyz"],
            K,
            T
        )



        if gt_uv is None:
            continue


        plt.scatter(
            gt_uv[0],
            gt_uv[1],
            c="lime",
            s=80,
            marker="o",
            label="GT"
        )


        if pred_uv is not None:


            plt.scatter(
                pred_uv[0],
                pred_uv[1],
                c="red",
                s=80,
                marker="x",
                label="Pred"
            )


            plt.plot(
                [
                    gt_uv[0],
                    pred_uv[0]
                ],
                [
                    gt_uv[1],
                    pred_uv[1]
                ],
                c="yellow"
            )


            plt.text(
                pred_uv[0],
                pred_uv[1],
                f"{error:.2f}m",
                color="yellow"
            )


    plt.axis("off")


    plt.savefig(
        save,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()



# ==========================================================
# BEV visualization
# ==========================================================

def draw_bev(
    matches,
    save
):

    plt.figure(
        figsize=(8,8)
    )


    for pred,gt,error in matches:


        gx=float(gt[0])

        gy=float(gt[1])


        px=float(pred["xyz"][0])

        py=float(pred["xyz"][1])



        plt.scatter(
            gx,
            gy,
            c="green",
            s=80,
            marker="o"
        )


        plt.scatter(
            px,
            py,
            c="red",
            s=80,
            marker="x"
        )


        plt.plot(
            [
                gx,
                px
            ],
            [
                gy,
                py
            ],
            c="yellow"
        )


        plt.text(
            px,
            py,
            f"{error:.2f}"
        )


    plt.xlabel(
        "X(m)"
    )

    plt.ylabel(
        "Y(m)"
    )


    plt.axis(
        "equal"
    )


    plt.grid()


    plt.legend(
        [
            "GT",
            "Prediction"
        ]
    )


    plt.savefig(
        save,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()



# ==========================================================
# Main
# ==========================================================

def main():


    args=parse_args()


    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    device=torch.device(
        "cuda"
    )


    split_file=find_sample(
        "data/splits/train.json",
        "data/splits/val.json",
        args.scene,
        args.frame_id
    )


    if split_file is None:

        raise RuntimeError(
            f"Cannot find {args.scene} frame {args.frame_id}"
        )


    print(
        "Loading:",
        split_file
    )



    dataset=SplitHeightNetDataset(
        split_file,
        image_size=(
            576,
            1024
        )
    )



    sample=None


    for i in range(
        len(dataset)
    ):

        item=dataset[i]


        if (
            str(item["frame_id"])
            ==
            str(args.frame_id)
        ):

            sample=item

            break



    if sample is None:

        raise RuntimeError(
            f"Frame {args.frame_id} not found"
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



    image=sample["image"].unsqueeze(0).to(device)

    K=sample["intrinsics"].to(device)

    T=sample["extrinsics_raw"].to(device)



    with torch.no_grad():


        output=model(
            image,
            K.unsqueeze(0),
            T.unsqueeze(0)
        )



    preds=decode_predictions(
        output["predictions"],
        args.score
    )



    matches=match_predictions(
        preds,
        sample["boxes_raw"],
        args.match_distance
    )



    print(
        "Frame:",
        sample["frame_id"]
    )


    print(
        "GT:",
        len(sample["boxes_raw"])
    )


    print(
        "Prediction:",
        len(preds)
    )


    print(
        "Matches:",
        len(matches)
    )



    for i,m in enumerate(matches):

        print(
            i,
            "error=",
            m[2]
        )



    rgb_path=os.path.join(
        args.output_dir,
        f"{args.scene}_{args.frame_id}_rgb.png"
    )


    bev_path=os.path.join(
        args.output_dir,
        f"{args.scene}_{args.frame_id}_bev.png"
    )


    K_vis = K.to("cpu")
    T_vis = T.to("cpu")
    draw_rgb(
        sample["image"].cpu(),
        K_vis,
        T_vis,
        matches,
        rgb_path
    )


    draw_bev(
        matches,
        bev_path
    )


    print(
        "Saved:"
    )

    print(
        rgb_path
    )

    print(
        bev_path
    )



if __name__=="__main__":

    main()