from .base import BaseModel

import torch
import torch.nn.functional as F


class LabelSmoothing(BaseModel):
    """
        Label smoothng with MFVI
    """
    def __init__(self, model, smoothing=0.10, lam_kl=1.0, 
            class_weight=None, mc_samples=32) -> None:
        
        super().__init__(model, class_weight, mc_samples)

        self.smoothing = smoothing
        self.lam_kl = lam_kl

        self.save_hyperparameters(ignore=['model'])

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

        # Predictive loss with label smoothing
        pred_loss = F.cross_entropy(torch.exp(y_pred), y, weight=self.class_weight, 
                        label_smoothing=self.smoothing)

        # KL Loss
        scaled_kl_loss = self.lam_kl * kl_loss

        # Total loss
        loss = pred_loss + scaled_kl_loss

        self.log('pred_loss', pred_loss.detach())
        self.log('kl_loss', kl_loss.detach())
        self.log('scaled_kl_loss', scaled_kl_loss.detach())

        return loss, y_pred

    def __repr__(self):
        return "LabelSmoothing" + \
                "\n    Model      : {}".format(self.model.__class__.__name__) + \
                "\n    lam_kl     : {:.3e}".format(self.lam_kl) + \
                "\n    class wts  : {}".format(self.class_weight) + \
                "\n    mc_samples : {}".format(self.mc_samples)
    
    @staticmethod
    def populate_missing_params(params, dataset):
        """
            Add lam_kl parameter if not provided
        """
        N = len(dataset)

        all_params = {
            'lam_kl': 1.0 / N
        }
        all_params.update(params)

        return all_params
