# Algorithms for training the model

from numpy import isin
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        ypred = []
        for i in range(self.mc_samples):
            _ypred, _kl = self(x)
            ypred.append(_ypred)
            kl_loss = _kl # KL loss is sample for all samples
        
        loss, ypred = self.compute_loss(ypred, y, kl_loss)
        preds = torch.argmax(ypred, dim=1)
        
        self.log('train_loss', loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        ypred = []
        for i in range(self.mc_samples):
            _ypred, _kl = self(x)
            ypred.append(_ypred)
            kl_loss = _kl # KL loss is sample for all samples
        
        val_loss, ypred = self.compute_loss(ypred, y, kl_loss)
        preds = torch.argmax(ypred, dim=1)
        
        self.log('val_loss', val_loss.detach())

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    # def setup(self, stage) -> None:
    #     self.class_weight = self.class_weight.to(self.device)
    #     return super().setup(stage)

    def compute_loss(self, ypred, y, kl_loss):
        """
            To be implemented in child
        """
        raise NotImplementedError('')



class MFVI(BaseModel):
    def __init__(self, model, lam_kl=1.0, class_weight=None, mc_samples=32) -> None:
        super().__init__(model, class_weight, mc_samples)
        self.lam_kl = lam_kl

    def compute_loss(self, y_pred, y, kl_loss):
        """
            Compute loss 

        y_pred  : tensor
            List (of length mc_samples) of Predicted log_softmax of shape (batch_size, classes)
        y       : tensor
            Target tensor of size (batch_size)
        kl_loss : tensor
            KL loss for forward.
        """

        if isinstance(y_pred, list):
            # In multiple MC samples are present, then find mean
            y_pred = torch.mean(torch.stack(y_pred), dim=0)
        if isinstance(kl_loss, list):
            kl_loss = torch.mean(torch.stack(kl_loss), dim=0)

        # Predictive loss
        pred_loss = F.nll_loss(y_pred, y, weight=self.class_weight)

        # KL Loss
        scaled_kl_loss = self.lam_kl * kl_loss

        # Total loss
        loss = pred_loss + scaled_kl_loss

        self.log('pred_loss', pred_loss.detach())
        self.log('kl_loss', kl_loss.detach())
        self.log('scaled_kl_loss', scaled_kl_loss.detach())

        return loss, y_pred


# For lookup
mfvi = MFVI
