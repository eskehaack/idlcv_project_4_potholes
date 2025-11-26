from model import ImageNetLightningModel
from dataloader import PotholesDataModule
from lightning import Trainer


def main():
    """
    Minimal example training loop.

    Adjust `data_dir` to point to an ImageNet-style folder with train/val(/test) splits.
    """
    data_dir = "potholes"
    datamodule = PotholesDataModule(data_dir=data_dir, batch_size=64, num_workers=0)
    model = ImageNetLightningModel(num_classes=2, lr=1e-3, pretrained=True)
    trainer = Trainer(max_epochs=10, accelerator="auto", devices="auto")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
