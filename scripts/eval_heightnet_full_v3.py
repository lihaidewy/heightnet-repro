import os
import argparse
import numpy as np

import torch
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from torch.utils.data import DataLoader

from src.heightnet.split_dataset import (
    SplitHeightNetDataset,
    heightnet_collate_fn
)

from src.heightnet.full_detector import (
    FullBEVHeightDetector
)



# ==========================================================
# Config
# ==========================================================

SCORE_THRESHOLD = 0.2

MATCH_DISTANCE = 2.0



# ==========================================================
# Args
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True
    )

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    return parser.parse_args()



# ==========================================================
# Model
# ==========================================================

def build_model(device):

    model = FullBEVHeightDetector(
        backbone_pretrained=False,
        backbone_depth=50,
        ground_z=-30
    )

    model=model.to(device)

    return model



def load_checkpoint(
    model,
    checkpoint
):

    ckpt=torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False
    )


    if "model" in ckpt:

        state=ckpt["model"]

    else:

        state=ckpt



    model.load_state_dict(
        state,
        strict=True
    )


    if "epoch" in ckpt:

        print(
            "Loaded Epoch:",
            ckpt["epoch"]
        )


    if "best_val" in ckpt:

        print(
            "Best val loss:",
            ckpt["best_val"]
        )



# ==========================================================
# Heatmap NMS
# ==========================================================

def local_peak_nms(
    heatmap,
    kernel=3
):

    maxpool=F.max_pool2d(
        heatmap,
        kernel,
        stride=1,
        padding=kernel//2
    )


    keep=(heatmap==maxpool)

    return heatmap*keep



# ==========================================================
# Circle NMS
# ==========================================================

def circle_nms(
    detections,
    radius=1.5
):

    if len(detections)==0:

        return []


    detections=sorted(
        detections,
        key=lambda x:x["score"],
        reverse=True
    )


    keep=[]


    for det in detections:

        valid=True


        for old in keep:

            dist=torch.norm(
                det["xyz"][:2]
                -
                old["xyz"][:2]
            )


            if dist < radius:

                valid=False

                break


        if valid:

            keep.append(det)


    return keep



# ==========================================================
# Decode one frame
# ==========================================================

def decode_single_prediction(
    prediction,
    threshold=SCORE_THRESHOLD
):


    heatmap=prediction["heatmap"].sigmoid()


    heatmap=local_peak_nms(
        heatmap
    )


    hm=heatmap[0,0]


    H,W=hm.shape


    scores,inds=torch.topk(
        hm.reshape(-1),
        k=min(100,hm.numel())
    )


    detections=[]


    for score,ind in zip(scores,inds):


        score=float(score)


        if score < threshold:

            continue



        y=ind//W

        x=ind%W



        dx=prediction["reg"][0,0,y,x]

        dy=prediction["reg"][0,1,y,x]



        X=40+x.float()+dx

        Y=-32+y.float()+dy



        z=prediction["height"][0,0,y,x]



        xyz=torch.stack(
            [
                X,
                Y,
                z-30
            ]
        )



        detections.append(
            {
                "xyz":xyz.detach().cpu(),
                "score":score
            }
        )



    return circle_nms(
        detections
    )
# ==========================================================
# Matching
# ==========================================================

def match_predictions(
    predictions,
    gt_boxes,
    threshold=MATCH_DISTANCE
):


    if len(predictions)==0:

        return [], [], list(range(len(gt_boxes)))



    if len(gt_boxes)==0:

        return [], list(range(len(predictions))), []



    pred_xy=torch.stack(
        [
            x["xyz"][:2]
            for x in predictions
        ]
    )


    gt_xy=gt_boxes[:,:2].cpu()



    distance=torch.cdist(
        pred_xy,
        gt_xy
    ).numpy()



    rows,cols=linear_sum_assignment(
        distance
    )


    matches=[]

    used_pred=set()

    used_gt=set()



    for r,c in zip(rows,cols):


        if distance[r,c] <= threshold:


            matches.append(
                (
                    predictions[r],
                    gt_boxes[c],
                    float(distance[r,c])
                )
            )


            used_pred.add(r)

            used_gt.add(c)



    fp=[
        i for i in range(len(predictions))
        if i not in used_pred
    ]


    fn=[
        i for i in range(len(gt_boxes))
        if i not in used_gt
    ]


    return matches, fp, fn



# ==========================================================
# Localization
# ==========================================================

def get_statistics(errors):


    if len(errors)==0:

        return {

            "count":0,
            "mean":float("nan"),
            "median":float("nan"),
            "p90":float("nan"),
            "acc05":float("nan"),
            "acc10":float("nan"),
            "acc20":float("nan")

        }



    errors=np.asarray(errors)



    return {

        "count":
            len(errors),

        "mean":
            float(errors.mean()),

        "median":
            float(np.median(errors)),

        "p90":
            float(np.percentile(errors,90)),

        "acc05":
            float(np.mean(errors<=0.5)),

        "acc10":
            float(np.mean(errors<=1.0)),

        "acc20":
            float(np.mean(errors<=2.0))

    }



# ==========================================================
# Camera distance
# ==========================================================

def camera_distance(
    point,
    world2cam
):

    device=world2cam.device


    point=point.to(device)



    homo=torch.cat(
        [
            point,
            torch.ones(
                1,
                device=device
            )
        ]
    )


    cam=world2cam @ homo


    return float(
        torch.norm(
            cam[:3]
        ).cpu()
    )



def distance_category(d):


    if d < 80:

        return "40-80m"


    elif d < 120:

        return "80-120m"


    else:

        return "120m+"



# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    model,
    loader,
    device
):


    TP=0

    FP=0

    FN=0



    errors=[]


    distance_errors={

        "40-80m":[],
        "80-120m":[],
        "120m+":[]

    }



    model.eval()



    with torch.no_grad():


        for step,batch in enumerate(loader):


            image=batch["image"].to(device)

            K=batch["intrinsics"].to(device)

            T=batch["extrinsics_raw"].to(device)



            output=model(
                image,
                K,
                T
            )



            batch_size=image.shape[0]



            for b in range(batch_size):


                pred_dict={}



                for k,v in output["predictions"].items():

                    pred_dict[k]=v[b:b+1]



                predictions=decode_single_prediction(
                    pred_dict,
                    SCORE_THRESHOLD
                )



                gt=batch["boxes_raw"][b]



                matches,fp,fn=match_predictions(
                    predictions,
                    gt
                )



                TP+=len(matches)

                FP+=len(fp)

                FN+=len(fn)



                for pred,gt_box,error in matches:


                    errors.append(
                        error
                    )



                    dist=camera_distance(
                        gt_box[:3],
                        T[b]
                    )


                    category=distance_category(
                        dist
                    )


                    distance_errors[category].append(
                        error
                    )



            if (step+1)%20==0:

                print(
                    f"Eval {step+1}/{len(loader)}"
                )



    return {

        "TP":TP,

        "FP":FP,

        "FN":FN,

        "errors":errors,

        "distance_errors":distance_errors

    }
# ==========================================================
# Print Result
# ==========================================================

def print_result(
    result
):


    TP=result["TP"]

    FP=result["FP"]

    FN=result["FN"]



    precision=TP/(TP+FP+1e-9)

    recall=TP/(TP+FN+1e-9)


    f1=(
        2*precision*recall
        /
        (precision+recall+1e-9)
    )


    mota=1-(FP+FN)/(TP+FN+1e-9)



    print("\n=================================")

    print(
        "Score threshold:",
        SCORE_THRESHOLD
    )


    print(
        "GT:",
        TP+FN
    )


    print(
        "TP:",
        TP
    )


    print(
        "FP:",
        FP
    )


    print(
        "FN:",
        FN
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


    print(
        "MOTA(no IDSW):",
        mota
    )



    print(
        "\n========== Localization =========="
    )


    stat=get_statistics(
        result["errors"]
    )


    print(
        "Count:",
        stat["count"]
    )


    print(
        "Mean:",
        stat["mean"]
    )


    print(
        "Median:",
        stat["median"]
    )


    print(
        "P90:",
        stat["p90"]
    )


    print(
        "<=0.5m:",
        stat["acc05"]
    )


    print(
        "<=1.0m:",
        stat["acc10"]
    )


    print(
        "<=2.0m:",
        stat["acc20"]
    )



    print(
        "\n========== Distance =========="
    )


    for key in [
        "40-80m",
        "80-120m",
        "120m+"
    ]:


        print("\n"+key)


        stat=get_statistics(
            result["distance_errors"][key]
        )


        print(
            "Count:",
            stat["count"]
        )

        print(
            "Mean:",
            stat["mean"]
        )

        print(
            "Median:",
            stat["median"]
        )

        print(
            "P90:",
            stat["p90"]
        )

        print(
            "<=0.5m:",
            stat["acc05"]
        )

        print(
            "<=1.0m:",
            stat["acc10"]
        )

        print(
            "<=2.0m:",
            stat["acc20"]
        )



# ==========================================================
# Main
# ==========================================================

def main():


    args=parse_args()



    device=torch.device(
        args.device
    )


    print(
        "Device:",
        device
    )


    dataset=SplitHeightNetDataset(
        args.split,
        image_size=(576,1024)
    )


    print(
        "Samples:",
        len(dataset)
    )



    loader=DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=heightnet_collate_fn
    )



    model=build_model(
        device
    )


    load_checkpoint(
        model,
        args.checkpoint
    )



    result=evaluate(
        model,
        loader,
        device
    )



    print_result(
        result
    )



if __name__=="__main__":

    main()