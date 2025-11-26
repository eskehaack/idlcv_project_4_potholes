import glob
import os

from utils.boundry_boxes import draw_boxes_on_image


def main(num_examples=3):
    """
    Load a few images and overlay their ground-truth bounding boxes.
    """
    if "magnus" not in os.getcwd():
        annotation_files = sorted(glob.glob("/dtu/datasets1/02516/potholes/annotations/**"))
        images_dir="/dtu/datasets1/02516/potholes/images"
    else:
        annotation_files = sorted(glob.glob("potholes/annotations/*.xml"))
        images_dir="potholes/images"

    if not annotation_files:
        raise RuntimeError("No annotation XML files found in 'potholes/annotations'.")

    for ann in annotation_files[:num_examples]:
        # use the provided utility, but point it to the local dataset
        draw_boxes_on_image(ann, images_dir=images_dir, color=(0, 255, 0), thickness=2, show=False, save_path="411_output.jpg")


if __name__ == "__main__":
    main()


