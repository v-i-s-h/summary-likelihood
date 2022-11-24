"""
    Training Determistic model with SGD+Momentum

    Warning: To be used only with Deterministic NNs
"""

from .base import BaseModel
from .sl import (
    AdaptiveSoftHistogram, SoftHistogram, MultiSoftHistogram,
    compute_sobs, compute_sobs_multi
)

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
        loss = self.compute_loss(ypred, y)

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

    def compute_loss(self, ypred, y):
        return F.nll_loss(ypred, y, weight=self.class_weight)

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


class SGDSLDeterministic(SGDDeterministic):
    def __init__(self, model, 
                sobs,
                bin_edges=None,
                alpha=1.0, 
                lam_sl=1.0, tau=0.0,
                class_weight=None) -> None:
        # Checks
        assert tau >= 1e-3 or tau == 0.0, "tau needs to be greater than 1e-3 (or 0.0 to disable annealing"

        super().__init__(model, class_weight)

        self.lam_sl = lam_sl
        
        self.register_buffer(name='sobs', tensor=torch.tensor(sobs))
        self.partitions = len(sobs)
        self.alpha = alpha
        self.tau = tau # annealing parameter
        
        # Histogram estimator
        if self.model.num_classes == 2:
            if bin_edges:
                print("INFO: Using AdaptiveSoftHistogram with binedges {}".format(bin_edges))
                print("INFO: sobs = {}".format(sobs))
                self.hist_est = AdaptiveSoftHistogram(bin_edges, sigma=500)
            else:
                print("INFO: Using equal bin histogram")
                print("INFO: sobs = {}".format(sobs))
                self.hist_est = SoftHistogram(bins=self.partitions, 
                            min=0, max=1, sigma=500).to(self.device)
        else:
            self.hist_est = MultiSoftHistogram(0.95, 0.85, 80).to(self.device)

        self.save_hyperparameters(ignore=['model'], logger=False)

    
    def compute_loss(self, y_pred, y):
        """
            Compute loss 

        y_pred  : tensor
            Predicted log_softmax of shape (batch_size, classes)
        y       : tensor
            Target tensor of size (batch_size)
        """

        # Predictive loss
        pred_loss = F.nll_loss(y_pred, y, weight=self.class_weight)

        
        # Calculate psuedo-observation likelihood
        yscore_samples = torch.exp(y_pred) # y_pred are log_soft of label 1
        yscore_hist = self.hist_est(yscore_samples)
        dirch_params = self.alpha * yscore_hist + 1e-4 # To avoid Dirich params -> 0
        
        # np.set_printoptions(threshold=np.inf, precision=3)
        # print(dirch_params.detach().cpu().numpy())
        # print("---------------", dirch_params.shape)
        
        ll_s_obs = torch.distributions.Dirichlet(dirch_params).log_prob(self.sobs)
        # raise RuntimeError()
        
        
        sl_loss = -1.0 * torch.mean(ll_s_obs) # mean over mc samples
        annlealed_scale = (
                np.exp(self.tau * (self.trainer.global_step + 1) / self.trainer.max_steps) - 1.0 + 1e-5
            ) / (
                np.exp(self.tau) - 1.0 + 1e-5
            ) * self.lam_sl
        scaled_sl_loss = annlealed_scale * sl_loss
        
        # Total loss
        loss = pred_loss + scaled_sl_loss

        self.log('pred_loss', pred_loss.detach())
        self.log('sl_loss', sl_loss.detach())
        self.log('sl_annealed_scale', annlealed_scale)
        self.log('scaled_sl_loss', scaled_sl_loss.detach())

        return loss

    def __repr__(self):
        return "SGD + SL Deterministic" + \
                "\n    Model      : {}".format(self.model.__class__.__name__) + \
                "\n    sobs       : {}".format(self.sobs) + \
                "\n    alpha      : {}".format(self.alpha) + \
                "\n    tau        : {}".format(self.tau) + \
                "\n    lam_sl     : {:.2e}".format(self.lam_sl) + \
                "\n    class wts  : {}".format(self.class_weight)
    
    @staticmethod
    def populate_missing_params(params, dataset):
        """
            Add following params if not provided
        """
        # To store computed params
        computed_params = {}

        # Get sobs
        if dataset.n_labels == 2:
            if 'adahist' in params:
                params['bin_edges'] = [
                    0.00, 0.01, 0.05, 0.10, 0.90, 0.95, 0.99, 1.00
                ]
            sobs = compute_sobs(params, dataset)
        else:
            sobs = compute_sobs_multi(params, dataset)
        computed_params['sobs'] = sobs
        if 'adahist' in params:
            computed_params['bin_edges'] = params['bin_edges']

        computed_params['alpha'] = params.get('alpha', 1.0)
        
        N = len(dataset)
        computed_params['lam_sl'] = params.get('lam_sl', 1.0/N)
        computed_params['tau'] = params.get('tau', 0.0)
        
        return computed_params
