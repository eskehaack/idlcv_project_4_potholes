import argparse
import glob
import os
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.read_xml import read_content
from utils.iou import compute_iou_matrix


def _default_paths() -> Tuple[str, str, str]:
    """
    Decide which root paths to use depending on environment.

    Returns
    -------
    annotations_dir : str
    proposals_dir   : str
    images_dir      : str
    """
    if "magnus" not in os.getcwd():
        annotations_dir = "potholes/annotations"
        images_dir = "potholes/images"
    else:
        annotations_dir = os.path.join("potholes", "annotations")
        images_dir = os.path.join("potholes", "images")

    proposals_dir = os.path.join("P4_1", "proposals")
    return annotations_dir, proposals_dir, images_dir


def _xyxy_from_rects(rects: np.ndarray) -> np.ndarray:
    """
    Convert rects from (x, y, w, h) to (xmin, ymin, xmax, ymax).
    """
    return np.array(
        [[x, y, x + w, y + h] for (x, y, w, h) in rects],
        dtype=float,
    )


def _draw_boxes(
    img: np.ndarray,
    gt_boxes: List[List[float]],
    prop_boxes: np.ndarray,
    best_ious: np.ndarray,
    iou_thr: float,
) -> np.ndarray:
    """
    Draw GT boxes and proposals on a copy of the image.

    - GT boxes: green
    - Positive proposals (best IoU >= thr): red
    - Negative proposals: yellow
    """
    draw = img.copy()

    # Draw ground-truth boxes (green)
    for (xmin, ymin, xmax, ymax) in gt_boxes:
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))
        cv2.rectangle(draw, pt1, pt2, (0, 255, 0), 2)

    # For each proposal, find max IoU over GTs and color-code
    # best_ious is per-GT, so recompute per-proposal IoU max
    if len(gt_boxes) > 0 and len(prop_boxes) > 0:
        iou_matrix, _ = compute_iou_matrix(gt_boxes, prop_boxes)
        prop_max_ious = iou_matrix.max(axis=0)
    else:
        prop_max_ious = np.zeros(len(prop_boxes))

    for (box, iou_val) in zip(prop_boxes, prop_max_ious):
        xmin, ymin, xmax, ymax = box
        pt1 = (int(xmin), int(ymin))
        pt2 = (int(xmax), int(ymax))

        if iou_val >= iou_thr:
            color = (255, 0, 0)  # red for positive
        else:
            color = (0, 255, 255)  # yellow for negative

        cv2.rectangle(draw, pt1, pt2, color, 1)

    return draw


def inspect_single_image(
    ann_path: str,
    proposals_dir: str,
    images_dir: str,
    top_n: int = 200,
    iou_threshold: float = 0.5,
) -> None:
    """
    Visualize proposals and GT boxes for a single image.

    Parameters
    ----------
    ann_path : str
        Path to XML annotation file.
    proposals_dir : str
        Directory containing pre-computed proposal .npy files.
    images_dir : str
        Directory with the corresponding images.
    top_n : int
        Number of top proposals to visualize.
    iou_threshold : float
        IoU threshold for marking a proposal as positive.
    """
    filename, gt_boxes = read_content(ann_path)
    base = os.path.splitext(filename)[0]

    prop_path = os.path.join(proposals_dir, f"{base}_rects.npy")
    if not os.path.exists(prop_path):
        print(f"[WARN] Missing proposals for {base}, expected at {prop_path}")
        return

    rects = np.load(prop_path)
    if rects.ndim == 1:
        rects = rects.reshape(-1, 4)

    prop_boxes = _xyxy_from_rects(rects)[:top_n]

    # Compute IoUs (per GT) for information
    _, best_ious_per_gt = compute_iou_matrix(gt_boxes, prop_boxes)
    mean_best_iou = best_ious_per_gt.mean() if len(best_ious_per_gt) > 0 else 0.0
    recall = (best_ious_per_gt >= iou_threshold).mean() if len(best_ious_per_gt) > 0 else 0.0

    print(f"Image: {filename}")
    print(f" - #GT boxes: {len(gt_boxes)}")
    print(f" - #Proposals (visualized): {len(prop_boxes)} (top {top_n})")
    print(f" - Mean best IoU (GT vs props): {mean_best_iou:.3f}")
    print(f" - Recall @ IoU >= {iou_threshold:.2f}: {recall:.3f}")

    # Load image and draw
    img_path = os.path.join(images_dir, filename)
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not read image at {img_path}")
        return

    drawn = _draw_boxes(img, gt_boxes, prop_boxes, best_ious_per_gt, iou_threshold)
    img_rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7))
    plt.title(f"{filename} | GT (green), +props (red), -props (yellow)")
    plt.axis("off")
    plt.imshow(img_rgb)
    plt.show()


def inspect_dataset(
    annotations_dir: str,
    proposals_dir: str,
    images_dir: str,
    top_n: int = 200,
    iou_threshold: float = 0.5,
    start_index: int = 0,
) -> None:
    """
    Step through the dataset, visualizing proposals + GT per image.
    """
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    if not ann_files:
        raise RuntimeError(f"No annotation XML files found in '{annotations_dir}'.")

    ann_files = ann_files[start_index:]

    for idx, ann_path in enumerate(ann_files, start=start_index):
        print(f"\n=== [{idx}] {ann_path} ===")
        inspect_single_image(
            ann_path=ann_path,
            proposals_dir=proposals_dir,
            images_dir=images_dir,
            top_n=top_n,
            iou_threshold=iou_threshold,
        )

        user_inp = input("Press <Enter> for next image, or type 'q' to quit: ").strip().lower()
        if user_inp == "q":
            break


def main():
    annotations_dir, proposals_dir, images_dir = _default_paths()

    parser = argparse.ArgumentParser(
        description="Inspect labeled proposals vs ground-truth for the potholes dataset."
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of top proposals to visualize per image.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a proposal as positive.",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=None,
        help="Optional index of a single image to visualize (0-based). "
             "If not provided, step through the dataset interactively.",
    )

    args = parser.parse_args()

    if args.image_index is not None:
        ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
        if not (0 <= args.image_index < len(ann_files)):
            raise IndexError(
                f"image_index {args.image_index} out of range [0, {len(ann_files) - 1}]"
            )

        ann_path = ann_files[args.image_index]
        inspect_single_image(
            ann_path=ann_path,
            proposals_dir=proposals_dir,
            images_dir=images_dir,
            top_n=args.top_n,
            iou_threshold=args.iou_threshold,
        )
    else:
        inspect_dataset(
            annotations_dir=annotations_dir,
            proposals_dir=proposals_dir,
            images_dir=images_dir,
            top_n=args.top_n,
            iou_threshold=args.iou_threshold,
        )


if __name__ == "__main__":
    main()


