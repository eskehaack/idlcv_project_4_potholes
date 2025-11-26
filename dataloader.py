from typing import Optional
import glob
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from lightning import LightningDataModule
from tqdm import tqdm


class CroppedProposalsDataset(Dataset):
    """
    Dataset wrapper that returns (image, label, origin, box)
    so that inference can map predictions back to the original
    image and bounding box coordinates.
    """

    def __init__(self, data: dict) -> None:
        self.images = data["images"]
        self.labels = data["labels"]
        self.boxes = data["boxes"]
        self.origins = data["origins"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.images[idx],
            self.labels[idx],
            self.origins[idx],
            self.boxes[idx],
        )


class PotholesDataModule(LightningDataModule):
    """
    Generic Potholes DataModule.

    Expects directory structure:
        data_dir/
          train/class_x/xxx.xml
          val/class_x/yyy.xml
          test/class_x/zzz.xml
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
        # target background fraction after *downsampling* the majority class
        self.downsample_bg_ratio = downsample_bg_ratio

        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Build / load cached cropped proposal datasets.

        We use the labeled proposals from `P4_1/labeled_proposals` and
        the corresponding original images in `potholes/images`.

        The first time this runs, it will:
          - Create an image-level split (80% train, 10% val, 10% test).
          - For each image, crop all proposals (up to 500) into
            ImageNet-sized patches and store them with labels.
          - Save three tensors: train.pt / val.pt / test.pt, each
            containing {"images": Tensor[N,3,H,W], "labels": Tensor[N]}.

        Subsequent runs simply load these cached tensors.
        """
        print("Setting up PotholesDataModule")
        cache_dir = os.path.join(self.data_dir, "cache")
        splits_dir = os.path.join(cache_dir, "splits")
        train_dir = os.path.join(splits_dir, "train")
        val_dir = os.path.join(splits_dir, "val")
        test_dir = os.path.join(splits_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        def _load_split_dir(split_dir: str):
            files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
            if not files:
                return None

            images_list = []
            labels_list = []
            boxes_list = []
            origins_list = []

            for fpath in files:
                data = torch.load(fpath)
                images_list.append(data["images"])
                labels_list.append(data["labels"])
                boxes_list.append(data["boxes"])
                origins_list.extend(data["origins"])

            images = torch.cat(images_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
            boxes = torch.cat(boxes_list, dim=0)

            return {
                "images": images,
                "labels": labels,
                "boxes": boxes,
                "origins": origins_list,
            }

        # If per-image cached files already exist, load them
        if all(
            glob.glob(os.path.join(d, "*.pt")) for d in (train_dir, val_dir, test_dir)
        ):
            print("Loading cached per-image crops from cache/splits")
            train_data = _load_split_dir(train_dir)
            val_data = _load_split_dir(val_dir)
            test_data = _load_split_dir(test_dir)

            if train_data is None or val_data is None or test_data is None:
                raise RuntimeError("Cached split directories are missing or empty.")

            self.train_dataset = CroppedProposalsDataset(train_data)
            self.val_dataset = CroppedProposalsDataset(val_data)
            self.test_dataset = CroppedProposalsDataset(test_data)
            return

        print("Building per-image crops from labeled proposals")
        # Build from labeled proposals
        labeled_dir = os.path.join("P4_1", "labeled_proposals")
        prop_files = sorted(glob.glob(os.path.join(labeled_dir, "*_props.npy")))
        if not prop_files:
            raise RuntimeError(f"No labeled proposals found in '{labeled_dir}'.")

        # Image-level split (based on base filenames)
        bases = [os.path.basename(f).replace("_props.npy", "") for f in prop_files]
        num_images = len(bases)
        indices = torch.randperm(num_images)
        n_train = int(0.8 * num_images)
        n_val = int(0.1 * num_images)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        def _process_single_image(idx: int, split_dir: str):
            base = bases[idx]
            img_path = os.path.join("potholes", "images", f"{base}.png")
            img = Image.open(img_path).convert("RGB")

            props = np.load(os.path.join(labeled_dir, f"{base}_props.npy"))
            labs = np.load(os.path.join(labeled_dir, f"{base}_labels.npy"))

            imgs = []
            labels = []
            boxes = []
            origins = []

            for box, lab in zip(props, labs):
                if lab == -1:
                    continue  # ignore ambiguous
                xmin, ymin, xmax, ymax = map(int, box)
                crop = img.crop((xmin, ymin, xmax, ymax))
                crop = self.eval_transforms(crop)
                imgs.append(crop)
                labels.append(int(lab))
                boxes.append([xmin, ymin, xmax, ymax])
                origins.append(base)

            if not imgs:
                return

            images_tensor = torch.stack(imgs, dim=0)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            boxes_tensor = torch.tensor(boxes, dtype=torch.float)
            origins_list = origins

            # Optional per-image downsampling towards desired bg/fg ratio
            if self.downsample_bg_ratio is not None:
                bg_mask = labels_tensor == 0
                fg_mask = labels_tensor == 1
                n_bg = int(bg_mask.sum())
                n_fg = int(fg_mask.sum())

                if n_bg > 0 and n_fg > 0:
                    target_bg_ratio = self.downsample_bg_ratio
                    target_fg_ratio = 1.0 - target_bg_ratio
                    target_ratio = target_bg_ratio / target_fg_ratio  # bg / fg

                    current_ratio = n_bg / n_fg

                    bg_indices = torch.nonzero(bg_mask, as_tuple=False).squeeze(1)
                    fg_indices = torch.nonzero(fg_mask, as_tuple=False).squeeze(1)

                    if current_ratio > target_ratio:
                        max_bg = int(target_ratio * n_fg)
                        perm = torch.randperm(n_bg)[:max_bg]
                        keep_bg = bg_indices[perm]
                        keep_fg = fg_indices
                    else:
                        max_fg = int(n_bg / target_ratio)
                        perm = torch.randperm(n_fg)[:max_fg]
                        keep_fg = fg_indices[perm]
                        keep_bg = bg_indices

                    keep_indices = torch.cat([keep_bg, keep_fg], dim=0)
                    perm_all = torch.randperm(len(keep_indices))
                    keep_indices = keep_indices[perm_all]

                    images_tensor = images_tensor[keep_indices]
                    labels_tensor = labels_tensor[keep_indices]
                    boxes_tensor = boxes_tensor[keep_indices]
                    origins_list = [origins_list[i] for i in keep_indices.tolist()]

            out_path = os.path.join(split_dir, f"{base}.pt")
            torch.save(
                {
                    "images": images_tensor,
                    "labels": labels_tensor,
                    "boxes": boxes_tensor,
                    "origins": origins_list,
                },
                out_path,
            )

        print("Processing train images")
        torch.manual_seed(42)
        for idx in tqdm(train_idx.tolist(), desc="Processing train images"):
            _process_single_image(idx, train_dir)
        print("Processing val images")
        for idx in tqdm(val_idx.tolist(), desc="Processing val images"):
            _process_single_image(idx, val_dir)
        print("Processing test images")
        for idx in tqdm(test_idx.tolist(), desc="Processing test images"):
            _process_single_image(idx, test_dir)
        print("Loading per-image files into memory for training")
        # Load per-image files into memory for training
        print("Loading train data")
        train_data = _load_split_dir(train_dir)
        print("Loading val data")
        val_data = _load_split_dir(val_dir)
        print("Loading test data")
        test_data = _load_split_dir(test_dir)

        if train_data is None or val_data is None or test_data is None:
            raise RuntimeError(
                "Failed to build one or more splits from labeled proposals."
            )

        self.train_dataset = CroppedProposalsDataset(train_data)
        self.val_dataset = CroppedProposalsDataset(val_data)
        self.test_dataset = CroppedProposalsDataset(test_data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
