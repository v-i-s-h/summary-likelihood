# Algorithms for training the model

import torch

from pytorch_lightning import LightningModule
import torchmetrics

class BaseModel(LightningModule):
    def __init__(self, model, class_weight=None, mc_samples=32) -> None:
        """
            model 
                Model for training
            class_weight : Tensor
                Tensor of size K
            lam : float
                Scaling for KL loss 
        """
        super().__init__()
        self.model = model
        if class_weight:
            self.register_buffer(name='class_weight', tensor=torch.tensor(class_weight))
        else:
            self.class_weight = None
        self.mc_samples = mc_samples

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_f1score = torchmetrics.F1Score(ignore_index=0)
        self.val_f1score = torchmetrics.F1Score(ignore_index=0)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        ypred = []
        for i in range(self.mc_samples):
            _ypred, _kl = self(x)
            ypred.append(_ypred)
            kl_loss = _kl # KL loss is same for all samples
        
        loss, ypred = self.compute_loss(ypred, y, kl_loss)
        preds = torch.argmax(ypred, dim=1)
        
        self.log('train_loss', loss.detach())

        # Compute metrics
        self.train_accuracy.update(preds, y)
        self.train_f1score.update(preds, y)

        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        self.log('train_f1', self.train_f1score, prog_bar=True)

        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = []
        for i in range(self.mc_samples):
            _ypred, _kl = self(x)
            ypred.append(_ypred)
            kl_loss = _kl # KL loss is same for all samples
        
        val_loss, ypred = self.compute_loss(ypred, y, kl_loss)
        preds = torch.argmax(ypred, dim=1)
        
        self.log('val_loss', val_loss.detach())

        # Compute metrics
        self.val_accuracy.update(preds, y)
        self.val_f1score.update(preds, y)

        return {
            'loss': val_loss,
            'ypred': ypred
        }

    def validation_epoch_end(self, outputs):

        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1score, prog_bar=True)

        # Log histogram of predictions
        ypred = torch.cat([o['ypred'] for o in outputs], dim=0)
        self.logger.experiment.add_histogram(
            'y_pred', torch.exp(ypred[:, 1]), # only for label 1
            global_step=self.current_epoch,
            bins=10)

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def compute_loss(self, ypred, y, kl_loss):
        """
            To be implemented in child
        """
        raise NotImplementedError('')