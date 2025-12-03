"""
Evaluate pothole detection performance across all images.

Computes:
- Recall: fraction of GT boxes that are "caught" by at least one prediction
- Precision: fraction of predictions that match a GT box
- F1 / Dice: harmonic mean of precision and recall
- Per-image breakdown and aggregate metrics

Optionally saves visualization images with GT and predicted boxes.
"""

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataloader import PotholesDataModule
from model import ImageNetLightningModel
from utils.iou import iou
from utils.read_xml import read_content
from utils.nms import non_max_suppression


def match_predictions_to_gt(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: List[List[int]],
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Match predicted boxes to ground truth boxes.

    Args:
        pred_boxes: (N, 4) array of predicted boxes [xmin, ymin, xmax, ymax]
        pred_scores: (N,) array of confidence scores
        gt_boxes: list of GT boxes [xmin, ymin, xmax, ymax]
        iou_threshold: IoU threshold for a match

    Returns:
        gt_matched: (num_gt,) bool array - True if GT was matched
        pred_matched: (num_pred,) bool array - True if prediction matched a GT
        pred_ious: (num_pred,) array - best IoU for each prediction
        tp: number of true positives (predictions matching a GT)
        fp: number of false positives (predictions not matching any GT)
        fn: number of false negatives (GTs not matched by any prediction)
    """
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)

    gt_matched = np.zeros(num_gt, dtype=bool)
    pred_matched = np.zeros(num_pred, dtype=bool)
    pred_ious = np.zeros(num_pred)

    if num_pred == 0 or num_gt == 0:
        return gt_matched, pred_matched, pred_ious, 0, num_pred, num_gt

    # Compute IoU matrix: (num_gt, num_pred)
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = iou(gt, pred.tolist())

    # Store best IoU for each prediction (for visualization)
    pred_ious = iou_matrix.max(axis=0) if num_gt > 0 else np.zeros(num_pred)

    # Greedy matching: for each GT, find best matching prediction
    for gt_idx in range(num_gt):
        best_pred_idx = np.argmax(iou_matrix[gt_idx])
        best_iou = iou_matrix[gt_idx, best_pred_idx]

        if best_iou >= iou_threshold and not pred_matched[best_pred_idx]:
            gt_matched[gt_idx] = True
            pred_matched[best_pred_idx] = True
            # Zero out this prediction's column to prevent re-matching
            iou_matrix[:, best_pred_idx] = 0

    tp = int(pred_matched.sum())
    fp = num_pred - tp
    fn = num_gt - int(gt_matched.sum())

    return gt_matched, pred_matched, pred_ious, tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute precision, recall, F1 from TP/FP/FN counts.
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_visualization(
    result: Dict,
    images_dir: str,
    output_path: str,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Save a visualization image showing GT boxes, predicted boxes with confidence,
    and metrics in the title.

    Args:
        result: dict with evaluation results for one image
        images_dir: directory containing original images
        output_path: path to save the visualization
    """
    image_id = result["image_id"]
    img_path = os.path.join(images_dir, f"{image_id}.png")

    if not os.path.exists(img_path):
        print(f"Warning: image not found at {img_path}")
        return

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: failed to load image at {img_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)

    # Draw GT boxes (green, thick)
    gt_boxes = result["gt_boxes"]
    gt_matched = result.get("gt_matched", np.ones(len(gt_boxes), dtype=bool))

    for i, gt in enumerate(gt_boxes):
        xmin, ymin, xmax, ymax = gt
        color = "lime" if gt_matched[i] else "red"  # Red for missed GTs
        linewidth = 3
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor=color, linewidth=linewidth, linestyle="-"
        )
        ax.add_patch(rect)
        # Label GT boxes
        label = "GT" if gt_matched[i] else "GT (MISSED)"
        ax.text(
            xmin, ymin - 5, label,
            color=color, fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )

    # Draw predicted boxes (blue/orange, with confidence)
    pred_boxes = result.get("pred_boxes", np.array([]))
    pred_scores = result.get("pred_scores", np.array([]))
    pred_matched = result.get("pred_matched", np.ones(len(pred_boxes), dtype=bool))

    for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
        xmin, ymin, xmax, ymax = box
        # Blue for TP, orange for FP
        color = "dodgerblue" if pred_matched[i] else "orange"
        linewidth = 2
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            fill=False, edgecolor=color, linewidth=linewidth, linestyle="--"
        )
        ax.add_patch(rect)
        # Show confidence score
        label = f"{score:.2f}"
        ax.text(
            xmax + 2, ymin + 10, label,
            color=color, fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7)
        )

    # Build title with metrics
    p = result["precision"]
    r = result["recall"]
    f1 = result["f1"]
    fn = result["fn"]
    tp = result["tp"]
    fp = result["fp"]
    num_gt = result["num_gt"]
    num_pred = result["num_pred_after_nms"]

    title = (
        f"{image_id}\n"
        f"Precision={p:.3f}  Recall={r:.3f}  F1/Dice={f1:.3f}\n"
        f"GT={num_gt}  Pred={num_pred}  TP={tp}  FP={fp}  FN={fn}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime", linewidth=3, label="GT (matched)"),
        Line2D([0], [0], color="red", linewidth=3, label="GT (missed)"),
        Line2D([0], [0], color="dodgerblue", linewidth=2, linestyle="--", label="Pred (TP)"),
        Line2D([0], [0], color="orange", linewidth=2, linestyle="--", label="Pred (FP)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_image(
    model: torch.nn.Module,
    dataset,
    image_id: str,
    sample_indices: List[int],
    gt_boxes: List[List[int]],
    device: torch.device,
    batch_size: int = 128,
    iou_threshold: float = 0.5,
    apply_nms: bool = True,
    nms_threshold: float = 0.3,
) -> Dict:
    """
    Evaluate model predictions for a single image.

    Returns dict with metrics and details.
    """
    if not sample_indices:
        # No proposals for this image
        return {
            "image_id": image_id,
            "num_gt": len(gt_boxes),
            "num_pred": 0,
            "num_pred_after_nms": 0,
            "tp": 0,
            "fp": 0,
            "fn": len(gt_boxes),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "missed_gt_indices": list(range(len(gt_boxes))),
            "pred_boxes": np.array([]),
            "pred_scores": np.array([]),
            "pred_matched": np.array([], dtype=bool),
            "gt_boxes": gt_boxes,
            "gt_matched": np.zeros(len(gt_boxes), dtype=bool),
        }

    # Run inference in batches
    all_boxes = []
    all_scores = []
    all_preds = []

    for batch_start in range(0, len(sample_indices), batch_size):
        batch_idxs = sample_indices[batch_start : batch_start + batch_size]

        batch_images = []
        batch_boxes = []

        for idx in batch_idxs:
            image, label, origin, box = dataset[idx]
            batch_images.append(image)
            batch_boxes.append(box)

        images_tensor = torch.stack(batch_images, dim=0).to(device)

        with torch.no_grad():
            logits = model(images_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu()
            # Use foreground probability as score
            fg_scores = probs[:, 1].cpu()

        for i, (pred, score, box) in enumerate(zip(preds, fg_scores, batch_boxes)):
            all_preds.append(int(pred))
            all_scores.append(float(score))
            all_boxes.append(box.numpy())

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_preds = np.array(all_preds)

    # Filter to foreground predictions only
    fg_mask = all_preds == 1
    pred_boxes = all_boxes[fg_mask]
    pred_scores = all_scores[fg_mask]

    num_pred_before_nms = len(pred_boxes)

    # Apply NMS to reduce overlapping predictions
    if apply_nms and len(pred_boxes) > 0:
        keep_indices = non_max_suppression(
            pred_boxes.tolist(), pred_scores.tolist(), iou_threshold=nms_threshold
        )
        pred_boxes = pred_boxes[keep_indices]
        pred_scores = pred_scores[keep_indices]

    num_pred_after_nms = len(pred_boxes)

    # Match predictions to GT
    gt_matched, pred_matched, pred_ious, tp, fp, fn = match_predictions_to_gt(
        pred_boxes, pred_scores, gt_boxes, iou_threshold=iou_threshold
    )

    metrics = compute_metrics(tp, fp, fn)

    # Find which GTs were missed
    missed_gt_indices = np.where(~gt_matched)[0].tolist()

    return {
        "image_id": image_id,
        "num_gt": len(gt_boxes),
        "num_pred": num_pred_before_nms,
        "num_pred_after_nms": num_pred_after_nms,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "missed_gt_indices": missed_gt_indices,
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "pred_matched": pred_matched,
        "gt_boxes": gt_boxes,
        "gt_matched": gt_matched,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pothole detection across all images."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching predictions to GT.",
    )
    parser.add_argument(
        "--nms_threshold",
        type=float,
        default=0.3,
        help="IoU threshold for NMS.",
    )
    parser.add_argument(
        "--no_nms",
        action="store_true",
        help="Disable NMS on predictions.",
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="potholes/annotations",
        help="Directory containing XML annotation files.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="potholes/images",
        help="Directory containing original images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save visualization images.",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save visualization images for each evaluated image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image results.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.ckpt}")
    model = ImageNetLightningModel.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # Set up data module
    datamodule = PotholesDataModule(
        data_dir="potholes", batch_size=args.batch_size, num_workers=0
    )
    datamodule.setup(stage=None)

    if args.split == "train":
        dataset = datamodule.train_dataset
    elif args.split == "val":
        dataset = datamodule.val_dataset
    else:
        dataset = datamodule.test_dataset

    samples = dataset.samples

    # Group samples by image
    image_to_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        image_to_indices[sample["origin"]].append(idx)

    image_ids = sorted(image_to_indices.keys())
    print(f"Evaluating {len(image_ids)} images in {args.split} split")

    # Load GT boxes for each image from annotations
    ann_files = glob.glob(os.path.join(args.annotations_dir, "*.xml"))
    image_to_gt: Dict[str, List[List[int]]] = {}
    for ann_path in ann_files:
        filename, gt_boxes = read_content(ann_path)
        base = os.path.splitext(filename)[0]
        image_to_gt[base] = gt_boxes

    # Create output directory if saving images
    if args.save_images:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving visualizations to: {args.output_dir}")

    # Evaluate each image
    all_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt, total_pred = 0, 0

    for image_id in tqdm(image_ids, desc="Evaluating images"):
        sample_indices = image_to_indices[image_id]
        gt_boxes = image_to_gt.get(image_id, [])

        result = evaluate_image(
            model=model,
            dataset=dataset,
            image_id=image_id,
            sample_indices=sample_indices,
            gt_boxes=gt_boxes,
            device=device,
            batch_size=args.batch_size,
            iou_threshold=args.iou_threshold,
            apply_nms=not args.no_nms,
            nms_threshold=args.nms_threshold,
        )

        all_results.append(result)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_gt += result["num_gt"]
        total_pred += result["num_pred_after_nms"]

        # Save visualization
        if args.save_images:
            output_path = os.path.join(args.output_dir, f"{image_id}.png")
            save_visualization(result, args.images_dir, output_path)

        if args.verbose:
            print(
                f"  {image_id}: GT={result['num_gt']}, Pred={result['num_pred_after_nms']}, "
                f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}, "
                f"P={result['precision']:.3f}, R={result['recall']:.3f}, F1={result['f1']:.3f}"
            )

    # Compute aggregate metrics
    aggregate = compute_metrics(total_tp, total_fp, total_fn)

    # Per-image averages
    per_image_precision = np.mean([r["precision"] for r in all_results])
    per_image_recall = np.mean([r["recall"] for r in all_results])
    per_image_f1 = np.mean([r["f1"] for r in all_results])

    # Find worst images (most missed GTs)
    results_sorted_by_fn = sorted(all_results, key=lambda x: x["fn"], reverse=True)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\nDataset: {args.split} split, {len(image_ids)} images")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"NMS: {'disabled' if args.no_nms else f'enabled (threshold={args.nms_threshold})'}")

    print(f"\n--- Aggregate Metrics (pooled across all images) ---")
    print(f"Total GT boxes:         {total_gt}")
    print(f"Total predictions:      {total_pred}")
    print(f"True Positives (TP):    {total_tp}")
    print(f"False Positives (FP):   {total_fp}")
    print(f"False Negatives (FN):   {total_fn} (missed GTs)")
    print(f"Precision:              {aggregate['precision']:.4f}")
    print(f"Recall (GT coverage):   {aggregate['recall']:.4f}")
    print(f"F1 / Dice Score:        {aggregate['f1']:.4f}")

    print(f"\n--- Per-Image Averages ---")
    print(f"Mean Precision:         {per_image_precision:.4f}")
    print(f"Mean Recall:            {per_image_recall:.4f}")
    print(f"Mean F1:                {per_image_f1:.4f}")

    print(f"\n--- Images with Most Missed GTs (FN) ---")
    for r in results_sorted_by_fn[:10]:
        if r["fn"] > 0:
            print(
                f"  {r['image_id']}: {r['fn']}/{r['num_gt']} GTs missed "
                f"(Recall={r['recall']:.2f})"
            )

    # Count images with perfect recall
    perfect_recall = sum(1 for r in all_results if r["recall"] == 1.0 and r["num_gt"] > 0)
    images_with_gt = sum(1 for r in all_results if r["num_gt"] > 0)
    print(f"\n--- Summary ---")
    print(f"Images with perfect recall: {perfect_recall}/{images_with_gt}")
    print(f"Images with zero recall:    {sum(1 for r in all_results if r['recall'] == 0.0 and r['num_gt'] > 0)}/{images_with_gt}")

    if args.save_images:
        print(f"\nVisualization images saved to: {args.output_dir}")

    return all_results, aggregate


if __name__ == "__main__":
    main()
