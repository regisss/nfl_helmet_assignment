import pandas as pd
import yaml
import numpy as np
import cv2

from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Dict


def get_yolo_format_bbox(image_width, image_height, bbox_left, bbox_width, bbox_top, bbox_height):
    """Return bounding box coordinates in YOLO format.

    Args:
        image_width (int): width of a frame.
        image_height (int): height of a frame.
        bbox_left (int): x-coordinate of the left corners of the bounding box.
        bbox_width (int): width of the bounding box.
        bbox_top (int): y-coordinate of the top corners of the bounding box.
        bbox_height (int): height of the bounding box.

    Returns:
        list[]: list containing the coordinates of the center of the bounding box, its width and its height (all normalized by the image dimensions).
    """

    xc = bbox_left + int(np.round(bbox_width / 2))  # xmin + width/2
    yc = bbox_top + int(np.round(bbox_height / 2))  # ymin + height/2

    return [
        xc / image_width,
        yc / image_height,
        bbox_width / image_width,
        bbox_height / image_height,
    ]


def make_training_validation_splits(path_to_images: Path, path_to_dataset_folder: Path, seed: int):
    """Split the dataset into a training set and a validation set.

    Args:
        path_to_images (Path): path toward original images.
        path_to_dataset_folder (Path): path where the dataset should be built.
        seed (int): random seed.

    Returns:
        Tuple[List[str]]: two lists of image names (one for training and another one for validation).
    """

    filenames = [filename for filename in path_to_images.iterdir()]

    # Split data into training and validation sets
    training_split, validation_split = train_test_split(filenames, test_size=0.2, random_state=seed)
    print(
        f"The training set contains {len(training_split)} images and the validation set {len(validation_split)}."
    )

    # The dataset is build according to yolov5 requirements (in terms of arborescence and folder names)
    for split, files in {"training": training_split, "validation": validation_split}.items():
        path_to_split_images = path_to_dataset_folder / "images" / split
        path_to_split_images.mkdir(parents=True, exist_ok=True)
        for filename in tqdm(files, desc=f"Copying {split} images"):
            (path_to_split_images / filename.name).symlink_to(Path.cwd() / filename)

    return training_split, validation_split


def write_labels(
    image_labels: pd.DataFrame,
    training_split: List[str],
    validation_split: List[str],
    label_to_class_id: Dict,
    path_to_dataset_folder: Path,
    path_to_images: Path,
):
    """Write labels and bounding-box coordinates for each image in the training and validation sets.

    Args:
        image_labels (pd.DataFrame): pandas dataframe containing bounding-box coordinates and labels.
        training_split (List[str]): list containing the names of all the images in the training set.
        validation_split (List[str]): list containing the names of all the images in the validation set.
        label_to_class_id (Dict): dictionary mapping labels to integers
        path_to_dataset_folder (Path): path toward the YOLOv5 dataset.
        path_to_images (Path): path toward all images (training + validation).
    """

    # Iterate over each image and write the labels and bbox coordinates to a .txt file.
    for filename in tqdm(list(path_to_images.iterdir()), desc="Writing labels for all images"):
        # Open image file to get the height and width
        img = cv2.imread(str(filename))
        image_height, image_width, _ = img.shape

        # Add bounding boxes and labels for the current image
        bboxes = []
        labels = []
        temp_dataframe = image_labels[image_labels["image"] == filename.name]
        for label, left, width, top, height in zip(
            temp_dataframe["label"],
            temp_dataframe["left"],
            temp_dataframe["width"],
            temp_dataframe["top"],
            temp_dataframe["height"],
        ):
            # get bbox in YOLO format
            yolo_bbox = get_yolo_format_bbox(image_width, image_height, left, width, top, height)
            bboxes.append(yolo_bbox)
            labels.append(label)

        # Determine the split
        split = None
        if filename in training_split:
            split = "training"
        elif filename in validation_split:
            split = "validation"
        path_to_labels_file = path_to_dataset_folder / f"labels/{split}" / f"{filename.stem}.txt"
        path_to_labels_file.parent.mkdir(parents=True, exist_ok=True)

        # Write text file
        with path_to_labels_file.open("w") as f:
            for label, bbox in zip(labels, bboxes):
                yolo_ground_truth = [str(i) for i in [label_to_class_id[label]] + bbox]
                yolo_ground_truth = " ".join(yolo_ground_truth)
                f.write(yolo_ground_truth)
                f.write("\n")


def build_dataset(path_to_kaggle_data: Path, seed: int = 27):
    """Build a dataset for YOLOv5.

    Args:
        path_to_kaggle_data (Path): path where Kaggle's data have been stored.
        seed (int, optional): Random seed. Defaults to 27.
    """

    path_to_images = path_to_kaggle_data / "images"  # path to JPG images
    path_to_labels = (
        path_to_kaggle_data / "image_labels.csv"
    )  # path to labels and bounding boxes coordinates

    # Read the CSV file
    image_labels = pd.read_csv(path_to_labels)

    print(f"The number of images is {len(image_labels.image.unique())}.")
    print(f"The number of bounding boxes is {len(image_labels)}.")

    # Dictionary that maps each label to an integer
    label_to_class_id = {label: i for i, label in enumerate(image_labels.label.unique())}
    print(f"Labels are {label_to_class_id}.")

    # Split the dataset into training and validation sets
    path_to_dataset_folder = Path("yolov5_dataset")
    training_split, validation_split = make_training_validation_splits(
        path_to_images, path_to_dataset_folder, seed
    )

    # Dataset configuration
    data_yaml = dict(
        train=f"../{path_to_dataset_folder}/images/training",
        val=f"../{path_to_dataset_folder}/images/validation",
        nc=len(label_to_class_id),  # number of classes in the dataset
        names=list(label_to_class_id),  # labels
    )

    # Note that the file is created in the yolov5/data/ directory.
    with open("yolov5/data/data.yaml", "w") as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)

    # for each image, write labels and bounding-box coordinates to a text file
    write_labels(
        image_labels,
        training_split,
        validation_split,
        label_to_class_id,
        path_to_dataset_folder,
        path_to_images,
    )
