# Train a Bayesian Neural Network with weight space prior using
# reparameterization


import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from models import LeNet
from datasets import BinaryMNISTC


class ClassifierModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = LeNet(2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # TODO: MC forwards

        y_pred, kl_loss = self(x)

        nll_loss = F.nll_loss(y_pred, y)

        loss = nll_loss + kl_loss

        self.log('nll_loss', nll_loss.detach())
        self.log('kl_loss', kl_loss.detach())
        self.log('train_loss', loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, kl_loss = self(x)
        nll_loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        loss = nll_loss + kl_loss

        self.log("val_loss", loss.detach())

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():

    tb_logger = pl_loggers.TensorBoardLogger("./zoo/")
    ckp_cb = ModelCheckpoint()

    model = ClassifierModel()

    # Dataset
    trainset = BinaryMNISTC('35', 'identity', 'train', transform=transforms.ToTensor())
    testset = BinaryMNISTC('35', 'identity', 'test', transform=transforms.ToTensor())
    train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    val_loader = DataLoader(testset, batch_size=1024, shuffle=False) # use test as val

    trainer = Trainer(
        max_steps=1000,
        gpus=0,
        logger=tb_logger,
        callbacks=[ckp_cb]
    )


    trainer.fit(model, train_loader, val_loader)


if __name__=="__main__":
    main()
