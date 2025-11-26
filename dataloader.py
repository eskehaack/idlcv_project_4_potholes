from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule

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
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

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
        if stage in (None, "fit"):
            self.train_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/train",
                transform=self.train_transforms,
            )
            self.val_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/val",
                transform=self.eval_transforms,
            )

        if stage in (None, "test"):
            self.test_dataset = datasets.ImageFolder(
                root=f"{self.data_dir}/test",
                transform=self.eval_transforms,
            )

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