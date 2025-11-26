import argparse
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

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
        default="potholes1",
        help="Base name of the image to visualize (without extension).",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=8,
        help="Maximum number of crop examples to show.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    model = ImageNetLightningModel.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # Set up data module and get test dataset
    datamodule = PotholesDataModule(data_dir="potholes", batch_size=64, num_workers=0)
    datamodule.setup(stage="test")
    test_ds = datamodule.test_dataset

    # Collect indices for the requested image_id
    origins: List[str] = test_ds.origins
    idxs = [i for i, o in enumerate(origins) if o == args.image_id]
    if not idxs:
        raise RuntimeError(
            f"No crops found for image_id='{args.image_id}' in test split."
        )

    # Stack all crops from this image
    images = torch.stack([test_ds.images[i] for i in idxs], dim=0).to(device)
    labels = torch.tensor(
        [test_ds.labels[i] for i in idxs], dtype=torch.long, device=device
    )
    boxes = torch.stack([test_ds.boxes[i] for i in idxs], dim=0)

    # Run model
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

    # Visualize some sample crops with predictions vs ground truth
    visualize_crops(images.cpu(), preds, labels.cpu(), max_examples=args.max_examples)

    # Visualize foreground boxes on the original image
    visualize_foreground_boxes(args.image_id, boxes, preds)


if __name__ == "__main__":
    main()
