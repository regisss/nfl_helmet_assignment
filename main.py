from pathlib import Path

from src.helmet_detection_yolov5.build_dataset import build_dataset


if __name__ == "__main__":
    path_to_kaggle_data = Path("data")

    build_dataset(path_to_kaggle_data)
