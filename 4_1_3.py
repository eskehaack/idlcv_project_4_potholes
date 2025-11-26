import glob
import os

import numpy as np
from tqdm import tqdm
from utils.read_xml import read_content
from utils.iou import evaluate_single_image, mabo_from_best_ious


def evaluate_proposals(
    annotations_dir="potholes/annotations",
    proposals_dir="P4_1/proposals",
    test_proposals=(100, 200, 300,400,500, 1000, 1500, 2000, 3000),
    iou_threshold=0.5,
):
    """
    Evaluate pre-computed region proposals against ground-truth boxes.

    Parameters
    ----------
    annotations_dir : str
        Directory with XML annotation files.
    proposals_dir : str
        Directory with saved proposal arrays (.npy) from 4_1_2.
    test_proposals : iterable of int
        Different numbers of top proposals N to evaluate.
    iou_threshold : float
        IoU threshold for counting a GT box as "covered".
    """
    ann_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
    if not ann_files:
        raise RuntimeError(f"No annotation XML files found in '{annotations_dir}'.")

    # For each N we will accumulate mean IoU, recall and all best-IoUs (for MABO)
    stats = {
        N: {
            "mean_best_ious": [],  # per-image mean
            "recall": [],          # per-image recall
            "all_best_ious": [],   # per-GT best IoUs for MABO
        }
        for N in test_proposals
    }

    for ann_path in tqdm(ann_files,desc="Evaluating proposals"):
        image_name, gt_boxes = read_content(ann_path)
        base = os.path.splitext(image_name)[0]
        prop_path = os.path.join(proposals_dir, f"{base}_rects.npy")

        if not os.path.exists(prop_path):
            print(f"Warning: missing proposals for {base}, expected at {prop_path}. Skipping.")
            continue

        rects = np.load(prop_path)
        # rects are (x, y, w, h) → convert to [xmin, ymin, xmax, ymax]
        prop_boxes = np.array(
            [[x, y, x + w, y + h] for (x, y, w, h) in rects],
            dtype=float,
        )

        # Use shared helper to evaluate this single image
        img_results = evaluate_single_image(gt_boxes, prop_boxes, test_proposals, iou_threshold)

        for N in test_proposals:
            stats[N]["mean_best_ious"].append(img_results[N]["mean_best_iou"])
            stats[N]["recall"].append(img_results[N]["recall"])
            # extend with this image's best IoU for every GT box
            stats[N]["all_best_ious"].extend(img_results[N]["best_ious"])

    print("Evaluation over dataset")
    print(f"(IoU threshold = {iou_threshold})")
    for N in test_proposals:
        if not stats[N]["mean_best_ious"]:
            print(f"N = {N:4d}: no images evaluated (missing proposals?)")
            continue

        mean_iou = np.mean(stats[N]["mean_best_ious"])
        mean_recall = np.mean(stats[N]["recall"])
        mabo = mabo_from_best_ious(stats[N]["all_best_ious"])
        print(
            f"Proposals: {N:4d} | "
            f"Mean best IoU (GT vs props): {mean_iou:.3f} | "
            f"Recall (IoU ≥ {iou_threshold:.2f}): {mean_recall:.3f} | "
            f"MABO: {mabo:.3f}"
        )


def main():
    if "magnus" not in os.getcwd():
        evaluate_proposals(annotations_dir="/dtu/datasets1/02516/potholes/annotations")
    else:
        evaluate_proposals()


if __name__ == "__main__":
    main()
