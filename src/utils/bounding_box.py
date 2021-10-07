import cv2
import pandas as pd
import numpy as np

from pathlib import Path
from typing import Tuple

# BGR value for displaying bounding boxes and labels
COLOR = (20, 11, 110)


def get_label_coordinates(image_width: int, left: int, top: int, height: int) -> Tuple[int, int]:
    """Label coordinates for display.

    Args:
        image_width (int): image width
        left (int): left coordinate
        top (int): top coordinate
        height (int): bounding box height

    Returns:
        Tuple[int, int]: label coordinates
    """

    # Define default coordinates
    coordinates = [left, top - 5]

    # Margin to assess whether coordinates are too close from the borders
    margin = 50

    # Coordinates are corrected if they are too close to the right border
    if np.abs(left - image_width) < margin:
        coordinates[0] -= 10

    # Coordinates are corrected if they are too close to the top border
    if np.abs(top - 5) < margin:
        coordinates[1] = top + height + 30

    return tuple(coordinates)


def draw_bounding_boxes(
    path_to_labels: Path, path_to_images: Path, path_to_output_folder: Path
) -> None:
    """Draw bounding boxes and labels in images.

    Args:
        path_to_labels (Path): path to the CSV file containing labels
        path_to_images (Path): path where images are stored
        path_to_output_folder (Path): path where annotated (bounding box + label) images will be written
    """

    # Create the output folder if needed
    path_to_output_folder.mkdir(parents=True, exist_ok=True)

    # Read the CSV file containing bounding boxes' labels
    bounding_box_labels = pd.read_csv(path_to_labels)
    # Drop bounding boxes related to sideline players
    bounding_box_labels = bounding_box_labels[bounding_box_labels["isSidelinePlayer"] == False]

    # Loop over images
    for file in path_to_images.iterdir():
        # Read image
        image = cv2.imread(str(file))

        # Get labels associated to the current image
        image_labels = bounding_box_labels[bounding_box_labels["video_frame"] == file.stem]

        # Loop over labels present in the current image
        for label, left, width, top, height in zip(
            image_labels["label"],
            image_labels["left"],
            image_labels["width"],
            image_labels["top"],
            image_labels["height"],
        ):
            # Draw rectangle corresponding to bounding box
            cv2.rectangle(image, (left, top), (left + width, top + height), COLOR)
            # Write label
            cv2.putText(
                image,
                label,
                get_label_coordinates(image.shape[1], left, top, height, width),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR,
                2,
            )
        # Write image
        cv2.imwrite(str(path_to_output_folder / file.name), image)


if __name__ == "__main__":
    path_to_labels = Path(
        "/home/regis/Documents/Projects/nfl_helmet_assignment/data/train_labels.csv"
    )
    path_to_images = Path("/home/regis/Documents/Projects/nfl_helmet_assignment/data/train_images")
    path_to_output_folder = Path(
        "/home/regis/Documents/Projects/nfl_helmet_assignment/data/train_images_bb"
    )

    draw_bounding_boxes(path_to_labels, path_to_images, path_to_output_folder)
