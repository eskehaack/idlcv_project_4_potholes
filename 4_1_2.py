import glob
import os

import numpy as np

from utils.selective_search import selective_search


def main(output_dir="P4_1/proposals", max_images=None, visualize_first=1):
    """
    Extract object proposals for all images in the potholes dataset using
    the provided selective_search utility.

    Parameters
    ----------
    output_dir : str
        Directory where the proposal arrays will be saved (.npy files).
    max_images : int or None
        If set, only process the first `max_images` images.
    visualize_first : int
        Number of images for which to display proposals (for sanity check).
    """
    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join("potholes", "images")
    if "magnus" not in os.getcwd():
        image_files = sorted(glob.glob("/dtu/datasets1/02516/potholes/images/**"))
    else:
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not image_files:
        raise RuntimeError(f"No images found in '{image_dir}'.")

    if max_images is not None:
        image_files = image_files[:max_images]

    for idx, img_path in enumerate(image_files):
        base = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Running selective search on {base} ({idx + 1}/{len(image_files)})")

        # Show proposals only for the first few images to keep it manageable
        show = idx < visualize_first
        rects = selective_search(img_path, show=show, save_path=None)

        save_path = os.path.join(output_dir, f"{base}_rects.npy")
        np.save(save_path, rects)
        print(f"Saved {len(rects)} proposals to {save_path}")


if __name__ == "__main__":
    main()


