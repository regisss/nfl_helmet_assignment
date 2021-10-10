import cv2

from pathlib import Path


def mp4_to_jpg(path_to_video: Path, path_to_image_folder: Path) -> None:
    """Write all the frames of a given mp4 file to a specified directory.

    Args:
        path_to_video (Path): path to the mp4 file
        path_to_image_folder (Path): path to the folder where frames will be written

    Raises:
        RuntimeError: if the video stream cannot be opened properly
    """

    # Create image folder if needed
    path_to_image_folder.mkdir(parents=True, exist_ok=True)

    # Get video
    video = cv2.VideoCapture(str(path_to_video))

    # Check if the video has been opened successfully
    if not video.isOpened():
        raise RuntimeError(f"Error opening the video stream at {path_to_video}")

    # Read video and write images
    i = 1
    while video.isOpened():
        # Get frame
        ret, frame = video.read()

        # If the frame was read correctly, write it to the destination folder
        if ret:
            cv2.imwrite(f"{path_to_image_folder / path_to_video.stem}_{i}.jpg", frame)
        else:
            break

        i += 1

    video.release()
    print(f"--- read video {path_to_video}")


def write_images_from_videos(path_to_video_folder: Path, path_to_image_folder: Path) -> None:
    """Write images from all the videos in the given folder.

    Args:
        path_to_video_folder (Path): path where all the videos to use are
        path_to_image_folder (Path): path where images will be written
    """

    print("Reading videos...")
    # Loop over files in the video folder
    for file in path_to_video_folder.iterdir():
        # Check if it is a mp4 file
        if file.suffix == ".mp4":
            # Write images
            mp4_to_jpg(file, path_to_image_folder)
    print("Reading videos done.")


if __name__ == "__main__":
    path_to_video_folder = Path("/home/regis/Documents/Projects/nfl_helmet_assignment/data/test")
    path_to_image_folder = Path(
        "/home/regis/Documents/Projects/nfl_helmet_assignment/data/test_images"
    )

    write_images_from_videos(path_to_video_folder, path_to_image_folder)
