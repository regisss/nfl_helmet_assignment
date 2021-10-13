import os
import yolov5.train

from pathlib import Path

from src.helmet_detection_yolov5.build_dataset import build_dataset


if __name__ == "__main__":
    path_to_kaggle_data = Path("data")

    # build_dataset(path_to_kaggle_data)

    # turn off W&B syncing if you don't need it
    os.environ["WANDB_MODE"] = "offline"

    yolov5.train.run(
        data="data.yaml",
        img=720,
        batch=32,
        epochs=1,
        weights="yolov5l.pt",
        project="models/helmet_detection_yolov5",
        name="run_",
        patience=5,
    )
