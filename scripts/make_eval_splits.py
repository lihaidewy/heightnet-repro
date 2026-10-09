import json
import os


# ==========================================================
# Config
# ==========================================================

S3S_FILE = "data/splits/train.json"
S3M_FILE = "data/splits/val.json"

OUTPUT_DIR = "data/splits"



# ==========================================================
# Load records
# ==========================================================

def load_json(path):

    with open(path, "r") as f:
        return json.load(f)



# ==========================================================
# Main
# ==========================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    train_records = load_json(
        S3S_FILE
    )

    val_records = load_json(
        S3M_FILE
    )


    all_records = (
        train_records
        +
        val_records
    )


    S3S_records = [
        x for x in all_records
        if x["scene"]=="S3S"
    ]


    S3M_records = [
        x for x in all_records
        if x["scene"]=="S3M"
    ]



    outputs = {

        "eval_scene3_all.json":
            all_records,

        "eval_S3S.json":
            S3S_records,

        "eval_S3M.json":
            S3M_records

    }



    for name,data in outputs.items():


        path=os.path.join(
            OUTPUT_DIR,
            name
        )


        with open(
            path,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )


        print(
            name
        )

        print(
            "Frames:",
            len(data)
        )

        print()



    print("======================")

    print(
        "Scene3:",
        len(all_records)
    )

    print(
        "S3S:",
        len(S3S_records)
    )

    print(
        "S3M:",
        len(S3M_records)
    )


if __name__=="__main__":

    main()