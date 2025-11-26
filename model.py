from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


class ImageNetLightningModel(LightningModule):


    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        self.model = backbone

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}








