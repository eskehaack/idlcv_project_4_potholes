import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm

from dataloader import PotholesDataModule
from model import ImageNetLightningModel


CLASS_NAMES = {0: "background", 1: "pothole"}


def tensor_to_np_image(t: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized CHW tensor to HWC uint8 for visualization.
    Assumes ImageNet normalization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    x = t * std + mean
    x = x.clamp(0, 1)
    x = (x.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return x


def visualize_crops(
    images: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    max_examples: int = 8,
) -> None:
    n = min(max_examples, images.shape[0])
    cols = min(4, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 4 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img_np = tensor_to_np_image(images[i])
        plt.imshow(img_np)
        p = int(preds[i])
        y = int(labels[i])
        title = f"Pred: {CLASS_NAMES.get(p, p)}\nGT: {CLASS_NAMES.get(y, y)}"
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_foreground_boxes(
    origin: str,
    boxes: torch.Tensor,
    preds: torch.Tensor,
    images_dir: str = "potholes/images",
) -> None:
    """
    Show the original image with only the boxes predicted as foreground.
    """
    img_path = os.path.join(images_dir, f"{origin}.png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    img = plt.imread(img_path)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    ax = plt.gca()

    for box, pred in zip(boxes, preds):
        if int(pred) != 1:
            continue
        xmin, ymin, xmax, ymax = box.tolist()
        w = xmax - xmin
        h = ymax - ymin
        rect = Rectangle(
            (xmin, ymin),
            w,
            h,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.title(f"Foreground proposals for {origin}")
    plt.axis("off")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize pothole detector crops and predictions."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to Lightning checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "--image_id",
        type=str,
        default=None,
        help="Base name of the image to visualize (without extension). "
        "If not provided, uses the first image in the split.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to draw the image from.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=8,
        help="Maximum number of crop examples to show.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    model = ImageNetLightningModel.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # Set up data module and select the requested split
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

    # The new dataset stores samples as a list of dicts
    samples = dataset.samples

    # Determine which image_id to use
    if args.image_id is None:
        # Use the first image in the split
        args.image_id = samples[0]["origin"]
        print(f"No --image_id provided, using first image: {args.image_id}")

    # Collect indices for the requested image_id
    idxs = [i for i, s in enumerate(samples) if s["origin"] == args.image_id]
    if not idxs:
        # List available image IDs
        available = sorted(set(s["origin"] for s in samples))
        raise RuntimeError(
            f"No crops found for image_id='{args.image_id}' in {args.split} split.\n"
            f"Available images: {available[:10]}{'...' if len(available) > 10 else ''}"
        )

    print(f"Found {len(idxs)} crops for image '{args.image_id}'")

    # Load crops on-the-fly and run inference in batches
    all_images = []
    all_labels = []
    all_boxes = []
    all_preds = []

    for batch_start in tqdm(
        range(0, len(idxs), args.batch_size), desc="Running inference"
    ):
        batch_idxs = idxs[batch_start : batch_start + args.batch_size]

        batch_images = []
        batch_labels = []
        batch_boxes = []

        for idx in batch_idxs:
            image, label, origin, box = dataset[idx]
            batch_images.append(image)
            batch_labels.append(label)
            batch_boxes.append(box)

        images_tensor = torch.stack(batch_images, dim=0).to(device)
        labels_tensor = torch.stack(batch_labels, dim=0)
        boxes_tensor = torch.stack(batch_boxes, dim=0)

        with torch.no_grad():
            logits = model(images_tensor)
            preds = logits.argmax(dim=1).cpu()

        all_images.append(images_tensor.cpu())
        all_labels.append(labels_tensor)
        all_boxes.append(boxes_tensor)
        all_preds.append(preds)

    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_boxes = torch.cat(all_boxes, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    # Print summary
    n_fg_pred = (all_preds == 1).sum().item()
    n_fg_gt = (all_labels == 1).sum().item()
    print(
        f"Predictions: {n_fg_pred} foreground, {len(all_preds) - n_fg_pred} background"
    )
    print(f"Ground truth: {n_fg_gt} foreground, {len(all_labels) - n_fg_gt} background")

    # Visualize some sample crops with predictions vs ground truth
    visualize_crops(all_images, all_preds, all_labels, max_examples=args.max_examples)

    # Visualize foreground boxes on the original image
    visualize_foreground_boxes(args.image_id, all_boxes, all_preds)


if __name__ == "__main__":
    main()
