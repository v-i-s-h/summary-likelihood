# Algorithms for training the model

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

        # TODO: MC forwards

        ypred, kl_loss = self(x)

        loss = self.compute_loss(ypred, y, kl_loss)

        self.log('train_loss', loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # TODO: Do MC forwards
        ypred, kl_loss = self(x)
        
        preds = torch.argmax(ypred, dim=1)

        val_loss = self.compute_loss(ypred, y, kl_loss)

        self.log("val_loss", val_loss.detach())

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
        pred_loss = F.nll_loss(y_pred, y, weight=self.class_weight)

        _kl_loss = self.lam_kl * kl_loss
        loss = pred_loss + _kl_loss

        self.log('pred_loss', pred_loss.detach())
        self.log('kl_loss', kl_loss.detach())
        self.log('kl_loss_eff', _kl_loss.detach())

        return loss


# For lookup
mfvi = MFVI
