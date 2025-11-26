import glob
import os

from utils.boundry_boxes import draw_boxes_on_image


def main(num_examples=3):
    """
    Load a few images and overlay their ground-truth bounding boxes.
    """
    annotation_files = sorted(glob.glob(os.path.join("potholes", "annotations", "*.xml")))
    if not annotation_files:
        raise RuntimeError("No annotation XML files found in 'potholes/annotations'.")

    for ann in annotation_files[:num_examples]:
        # use the provided utility, but point it to the local dataset
        draw_boxes_on_image(ann, images_dir="potholes/images", color=(0, 255, 0), thickness=2)


if __name__ == "__main__":
    main()


