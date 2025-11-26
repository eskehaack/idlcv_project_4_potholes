import glob
import os

import numpy as np
from tqdm import tqdm

from utils.read_xml import read_content
from utils.iou import compute_iou_matrix


def build_labeled_proposals(
    annotations_dir="potholes/annotations",
    proposals_dir="P4_1/proposals",
    output_dir="P4_1/labeled_proposals",
    num_proposals=2000,
    pos_iou_thr=0.5,
    neg_iou_thr=0.3,
):
    """
    Prepare a proposal dataset for training an object detector.

    For each image:
      - Load its region proposals (rects) from `proposals_dir`.
      - Keep the first `num_proposals` (default: 2000).
      - Convert rects (x, y, w, h) -> boxes [xmin, ymin, xmax, ymax].
      - Compute IoU with GT boxes and assign labels:
          1 = foreground (IoU >= pos_iou_thr)
          0 = background (IoU <= neg_iou_thr)
        Boxes with IoU in (neg_iou_thr, pos_iou_thr) are ignored.
      - Save proposals and labels as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    if not ann_files:
        raise RuntimeError(f"No annotation XML files found in '{annotations_dir}'.")

    for ann_path in tqdm(ann_files, desc="Building labeled proposals"):
        image_name, gt_boxes = read_content(ann_path)
        base = os.path.splitext(image_name)[0]

        prop_path = os.path.join(proposals_dir, f"{base}_rects.npy")
        if not os.path.exists(prop_path):
            print(f"Warning: missing proposals for {base}, expected at {prop_path}. Skipping.")
            continue

        rects = np.load(prop_path)
        if rects.size == 0:
            print(f"Warning: no proposals for {base}. Skipping.")
            continue

        # Keep only first num_proposals
        rects = rects[:num_proposals]

        # Convert (x, y, w, h) → [xmin, ymin, xmax, ymax]
        prop_boxes = np.array(
            [[x, y, x + w, y + h] for (x, y, w, h) in rects],
            dtype=float,
        )

        # For each proposal, we want max IoU against any GT
        # compute_iou_matrix gives best IoU per GT, so we recompute by swapping roles
        _, best_ious_per_prop = compute_iou_matrix(prop_boxes, gt_boxes)

        labels = np.full(len(prop_boxes), -1, dtype=int)  # -1 = ignore
        labels[best_ious_per_prop >= pos_iou_thr] = 1
        labels[best_ious_per_prop <= neg_iou_thr] = 0

        # Save per-image arrays
        props_out = os.path.join(output_dir, f"{base}_props.npy")
        labels_out = os.path.join(output_dir, f"{base}_labels.npy")
        np.save(props_out, prop_boxes)
        np.save(labels_out, labels)


def main():
    build_labeled_proposals()


if __name__ == "__main__":
    main()


