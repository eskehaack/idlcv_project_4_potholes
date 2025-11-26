from typing import Optional, List, Tuple
import glob
import os
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from lightning import LightningDataModule
from tqdm import tqdm


class ProposalCropDataset(Dataset):
    """
    Lazy-loading dataset for proposal crops.

    Stores only metadata (image path, box, label) and crops on-the-fly
    in __getitem__. This is the standard approach used by R-CNN implementations
    like Detectron2.

    Each sample is loaded from a lightweight index file that contains:
      - image_path: path to the original image
      - box: [xmin, ymin, xmax, ymax]
      - label: 0 or 1
    """

    def __init__(self, index_file: str, transform=None) -> None:
        """
        Args:
            index_file: Path to JSON file containing list of
                        {"image_path": str, "box": [x1,y1,x2,y2], "label": int}
            transform: torchvision transform to apply to cropped PIL image
        """
        with open(index_file, "r") as f:
            self.samples = json.load(f)
        self.transform = transform

        # Cache for the last loaded image to avoid re-reading when
        # consecutive samples come from the same image
        self._cached_image_path: Optional[str] = None
        self._cached_image: Optional[Image.Image] = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        box = sample["box"]
        label = sample["label"]
        origin = sample["origin"]

        # Load image (with simple caching for consecutive same-image accesses)
        if self._cached_image_path != image_path:
            self._cached_image = Image.open(image_path).convert("RGB")
            self._cached_image_path = image_path

        # Crop the region
        xmin, ymin, xmax, ymax = box
        crop = self._cached_image.crop((xmin, ymin, xmax, ymax))

        # Apply transform
        if self.transform is not None:
            crop = self.transform(crop)

        label_tensor = torch.tensor(label, dtype=torch.long)
        box_tensor = torch.tensor(box, dtype=torch.float)

        return crop, label_tensor, origin, box_tensor


class PotholesDataModule(LightningDataModule):
    """
    Potholes DataModule using lazy crop-on-the-fly loading.

    Preprocessing creates lightweight JSON index files containing
    (image_path, box, label) tuples. Training loads images and crops
    on-the-fly, which is memory-efficient and fast.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        img_size: int = 224,
        downsample_bg_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.downsample_bg_ratio = downsample_bg_ratio

        # Training uses augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Eval uses deterministic resize
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build or load index files for train/val/test splits.

        Index files are small JSON files containing metadata only:
        no images or tensors are stored, making this very fast.
        """
        print("Setting up PotholesDataModule")
        cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        train_index = os.path.join(cache_dir, "train_index.json")
        val_index = os.path.join(cache_dir, "val_index.json")
        test_index = os.path.join(cache_dir, "test_index.json")

        # If index files exist, just load them
        if os.path.exists(train_index) and os.path.exists(val_index) and os.path.exists(test_index):
            print("Loading existing index files from cache")
            self.train_dataset = ProposalCropDataset(train_index, self.train_transforms)
            self.val_dataset = ProposalCropDataset(val_index, self.eval_transforms)
            self.test_dataset = ProposalCropDataset(test_index, self.eval_transforms)
            print(f"  Train: {len(self.train_dataset)} crops")
            print(f"  Val:   {len(self.val_dataset)} crops")
            print(f"  Test:  {len(self.test_dataset)} crops")
            return

        print("Building index files from labeled proposals")

        # Load labeled proposals
        labeled_dir = os.path.join("P4_1", "labeled_proposals")
        prop_files = sorted(glob.glob(os.path.join(labeled_dir, "*_props.npy")))
        if not prop_files:
            raise RuntimeError(f"No labeled proposals found in '{labeled_dir}'.")

        # Image-level split
        bases = [os.path.basename(f).replace("_props.npy", "") for f in prop_files]
        num_images = len(bases)

        torch.manual_seed(42)
        indices = torch.randperm(num_images).tolist()
        n_train = int(0.8 * num_images)
        n_val = int(0.1 * num_images)

        train_bases = [bases[i] for i in indices[:n_train]]
        val_bases = [bases[i] for i in indices[n_train:n_train + n_val]]
        test_bases = [bases[i] for i in indices[n_train + n_val:]]

        def _build_index(base_list: List[str], is_train: bool) -> List[dict]:
            """Build list of sample dicts for given image bases."""
            samples = []

            for base in tqdm(base_list, desc="Building index"):
                img_path = os.path.join("potholes", "images", f"{base}.png")
                if not os.path.exists(img_path):
                    print(f"Warning: image not found: {img_path}")
                    continue

                props = np.load(os.path.join(labeled_dir, f"{base}_props.npy"))
                labs = np.load(os.path.join(labeled_dir, f"{base}_labels.npy"))

                # Collect valid (non-ambiguous) proposals
                fg_samples = []
                bg_samples = []

                for box, lab in zip(props, labs):
                    if lab == -1:
                        continue  # skip ambiguous
                    xmin, ymin, xmax, ymax = map(int, box)
                    sample = {
                        "image_path": img_path,
                        "box": [xmin, ymin, xmax, ymax],
                        "label": int(lab),
                        "origin": base,
                    }
                    if lab == 1:
                        fg_samples.append(sample)
                    else:
                        bg_samples.append(sample)

                # Downsample background if needed (only for training)
                if is_train and self.downsample_bg_ratio is not None:
                    n_fg = len(fg_samples)
                    n_bg = len(bg_samples)

                    if n_fg > 0 and n_bg > 0:
                        target_ratio = self.downsample_bg_ratio / (1.0 - self.downsample_bg_ratio)
                        current_ratio = n_bg / n_fg

                        if current_ratio > target_ratio:
                            max_bg = int(target_ratio * n_fg)
                            np.random.seed(42)
                            indices_keep = np.random.permutation(n_bg)[:max_bg]
                            bg_samples = [bg_samples[i] for i in indices_keep]

                samples.extend(fg_samples)
                samples.extend(bg_samples)

            return samples

        print("Building train index...")
        train_samples = _build_index(train_bases, is_train=True)
        print("Building val index...")
        val_samples = _build_index(val_bases, is_train=False)
        print("Building test index...")
        test_samples = _build_index(test_bases, is_train=False)

        # Save index files
        with open(train_index, "w") as f:
            json.dump(train_samples, f)
        with open(val_index, "w") as f:
            json.dump(val_samples, f)
        with open(test_index, "w") as f:
            json.dump(test_samples, f)

        print(f"Saved index files:")
        print(f"  Train: {len(train_samples)} crops -> {train_index}")
        print(f"  Val:   {len(val_samples)} crops -> {val_index}")
        print(f"  Test:  {len(test_samples)} crops -> {test_index}")

        # Create datasets
        self.train_dataset = ProposalCropDataset(train_index, self.train_transforms)
        self.val_dataset = ProposalCropDataset(val_index, self.eval_transforms)
        self.test_dataset = ProposalCropDataset(test_index, self.eval_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
