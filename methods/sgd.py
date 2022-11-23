"""
    Training Determistic model with SGD+Momentum

    Warning: To be used only with Deterministic NNs
"""

from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch_lightning import LightningModule
import torchmetrics



class SGDDeterministic(LightningModule):
    def __init__(self, model,
            class_weight=None) -> None:

        super().__init__()
        self.model = model
        if class_weight:
            self.register_buffer(name='class_weight', tensor=torch.tensor(class_weight))
        else:
            self.class_weight = None

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

        if self.model.num_classes == 2:
            ignore_index = 0
        else:
            ignore_index = None
        self.train_f1score = torchmetrics.F1Score(ignore_index=ignore_index)
        self.val_f1score = torchmetrics.F1Score(ignore_index=ignore_index)
        self.test_f1score = torchmetrics.F1Score(ignore_index=ignore_index)

        # self.train_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')
        self.val_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')
        self.test_ece = torchmetrics.CalibrationError(n_bins=10, norm='l1')    
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        ypred = self(x)

        # Compute loss
        loss = F.nll_loss(ypred, y, weight=self.class_weight)

        preds = torch.argmax(ypred, dim=1)

        self.log('train_loss', loss.detach())

        self.train_accuracy.update(preds, y)
        self.train_f1score.update(preds, y)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        self.log('train_f1', self.train_f1score, prog_bar=True)
        # self.log('train_ece', self.train_ece, prog_bar=True)

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = self(x)
        
        val_loss = F.nll_loss(ypred, y, weight=self.class_weight)

        preds = torch.argmax(ypred, dim=1)
        pred_prob = torch.exp(ypred)
        
        self.log('val_loss', val_loss.detach())

        self.val_accuracy.update(preds, y)
        self.val_f1score.update(preds, y)
        self.val_ece.update(pred_prob, y)

        return {
            'loss': val_loss,
            'ypred': ypred
        }

    def validation_epoch_end(self, outputs):

        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1score, prog_bar=True)
        self.log('val_ece', self.val_ece, prog_bar=True)

        # Log histogram of predictions
        ypred = torch.cat([o['ypred'] for o in outputs], dim=0)
        if self.model.num_classes == 2:
            # For binary classification, only log score for class 1
            self.logger.experiment.add_histogram(
                'y_pred_val', torch.exp(ypred[:, 1]), # only for label 1
                global_step=self.current_epoch,
                bins=10)
        else:
            for i in range(self.model.num_classes):
                self.logger.experiment.add_histogram(
                    'y_pred_val_{:02d}'.format(i), 
                    torch.exp(ypred[:, i]), # for label `i`
                    global_step=self.current_epoch,
                    bins=10)

        return None

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = self(x)
        
        test_loss = F.nll_loss(ypred, y, weight=self.class_weight)
        preds = torch.argmax(ypred, dim=1)
        pred_prob = torch.exp(ypred)
        
        self.log('test_loss', test_loss.detach())

        # Compute metrics
        self.test_accuracy.update(preds, y)
        self.test_f1score.update(preds, y)
        self.test_ece.update(pred_prob, y)

        return {
            'loss': test_loss,
            'ypred': ypred
        }

    def test_epoch_end(self, outputs):

        test_acc = self.test_accuracy.compute()
        test_f1 = self.test_f1score.compute()
        test_ece = self.test_ece.compute()

        self.log('test_acc', test_acc, prog_bar=True)
        self.log('test_f1', test_f1, prog_bar=True)
        self.log('test_ece', test_ece, prog_bar=True)

        # Log histogram of predictions
        ypred = torch.cat([o['ypred'] for o in outputs], dim=0)
        if self.model.num_classes == 2:
            # For binary classification, only log score for class 1
            self.logger.experiment.add_histogram(
                'y_pred_test', torch.exp(ypred[:, 1]), # only for label 1
                global_step=self.current_epoch,
                bins=10)
        else:
            for i in range(self.model.num_classes):
                self.logger.experiment.add_histogram(
                    'y_pred_test_{:02d}'.format(i), 
                    torch.exp(ypred[:, i]), # for label `i`
                    global_step=self.current_epoch,
                    bins=10)

        return None

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.90)
        # return torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(self.parameters(), 
                        lr=1e-2, momentum=0.9, 
                        weight_decay=5e-4, 
                        nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                        mode='min', factor=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "frequency": self.trainer.check_val_every_n_epoch
            }
        }

    def __repr__(self):
        return "SGD Deterministic" + \
                "\n    Model      : {}".format(self.model.__class__.__name__) + \
                "\n    class wts  : {}".format(self.class_weight)
    
    @staticmethod
    def populate_missing_params(params, dataset):
        """
            Add following params if not provided
        """
        # To store computed params
        computed_params = {}

        # Nothing to add for SGD + Deterministic
        
        return computed_params

