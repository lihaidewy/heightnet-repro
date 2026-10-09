import json
import os


OUTPUT_DIR = "data/splits"


def collect_scene_records():

    records = []


    # S3S
    for frame_id in range(0, 10000):

        pass



def find_all_frames():

    """
    从已有split文件中读取所有可能记录，
    去重后生成完整Scene3评估列表
    """

    files = [

        "data/splits/train.json",
        "data/splits/val.json",
        "data/splits/excluded.json"

    ]


    all_records=[]


    for file in files:

        if not os.path.exists(file):

            print(
                "Missing:",
                file
            )

            continue


        with open(file,"r") as f:

            data=json.load(f)


        all_records.extend(data)



    # 去重

    unique={}


    for x in all_records:

        key=(
            x["scene"],
            str(x["frame_id"])
        )

        unique[key]=x



    records=list(
        unique.values()
    )


    return records



def main():


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    records=find_all_frames()



    s3s=[
        x for x in records
        if x["scene"]=="S3S"
    ]


    s3m=[
        x for x in records
        if x["scene"]=="S3M"
    ]



    outputs={

        "eval_scene3_all.json":
            records,

        "eval_S3S.json":
            s3s,

        "eval_S3M.json":
            s3m

    }



    for name,data in outputs.items():


        path=os.path.join(
            OUTPUT_DIR,
            name
        )


        with open(path,"w") as f:

            json.dump(
                data,
                f,
                indent=2
            )


        print(
            name,
            "frames:",
            len(data)
        )



    print("====================")

    print(
        "Scene3:",
        len(records)
    )

    print(
        "S3S:",
        len(s3s)
    )

    print(
        "S3M:",
        len(s3m)
    )



if __name__=="__main__":

    main()