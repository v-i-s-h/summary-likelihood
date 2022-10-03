from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.stats as st
from scipy.special import betainc
from scipy.optimize import minimize


def compute_sobs(params, dataset):
    """
        Compute the s_obs from the label frequency
    """
    _, f = np.unique(
        [dataset.ds.dataset.targets[i] for i in dataset.ds.indices], 
        return_counts=True)
    f = f / f.sum()

    return f



class SummaryLikelihoodIm(BaseModel):
    def __init__(self, model,
            sobs, alpha=1.0,
            lam_kl=1.0, lam_sl=1.0, tau=0.0,
            class_weight=None, mc_samples=32) -> None:
        # Checks
        assert tau >= 1e-3 or tau == 0.0, "tau needs to be greater than 1e-3 (or 0.0 to disable annealing"

        super().__init__(model, class_weight, mc_samples)

        self.lam_kl = lam_kl
        self.lam_sl = lam_sl
        
        self.register_buffer(name='sobs', tensor=torch.tensor(sobs))
        self.partitions = len(sobs)
        self.alpha = alpha
        self.tau = tau # annealing parameter

        self.save_hyperparameters(ignore=['model'], logger=False)


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

        mc_y_pred = y_pred

        if isinstance(y_pred, list):
            # If multiple MC samples are present, then find mean
            y_pred = torch.mean(torch.stack(y_pred), dim=0)
        if isinstance(kl_loss, list):
            kl_loss = torch.mean(torch.stack(kl_loss), dim=0)

        # Predictive loss
        pred_loss = F.nll_loss(y_pred, y, weight=self.class_weight)

        # KL Loss
        scaled_kl_loss = self.lam_kl * kl_loss

        # Calculate psuedo-observation likelihood
        mc_y_pred = torch.stack(mc_y_pred)
        yscore_samples = torch.exp(mc_y_pred)
        yscore_hist = self.hist_est(yscore_samples)
        dirch_params = self.alpha * yscore_hist + 1e-4 # To avoid Dirich params -> 0
        
        ll_s_obs = torch.distributions.Dirichlet(dirch_params).log_prob(self.sobs)
        
        sl_loss = -1.0 * torch.mean(ll_s_obs) # mean over mc samples
        annlealed_scale = (
                np.exp(self.tau * (self.trainer.global_step + 1) / self.trainer.max_steps) - 1.0 + 1e-5
            ) / (
                np.exp(self.tau) - 1.0 + 1e-5
            ) * self.lam_sl
        scaled_sl_loss = annlealed_scale * sl_loss
        
        # Total loss
        loss = pred_loss + scaled_sl_loss + scaled_kl_loss

        self.log('pred_loss', pred_loss.detach())
        self.log('kl_loss', kl_loss.detach())
        self.log('scaled_kl_loss', scaled_kl_loss.detach())
        self.log('sl_loss', sl_loss.detach())
        self.log('sl_annealed_scale', annlealed_scale)
        self.log('scaled_sl_loss', scaled_sl_loss.detach())

        return loss, y_pred

    def hist_est(self, yscores):
        """
            Estimate soft histogram
        """
        # Take hist as sample mean over softmax scores
        score_hist = torch.mean(yscores, dim=1)

        return score_hist

    def __repr__(self):
        return "SL" + \
                "\n    Model      : {}".format(self.model.__class__.__name__) + \
                "\n    sobs       : {}".format(self.sobs) + \
                "\n    alpha      : {}".format(self.alpha) + \
                "\n    tau        : {}".format(self.tau) + \
                "\n    lam_kl     : {:.2e}".format(self.lam_kl) + \
                "\n    lam_sl     : {:.2e}".format(self.lam_sl) + \
                "\n    class wts  : {}".format(self.class_weight) + \
                "\n    mc_samples : {}".format(self.mc_samples)
    
    @staticmethod
    def populate_missing_params(params, dataset):
        """
            Add following params if not provided
        """
        # To store computed params
        computed_params = {}

        # Get sobs
        sobs = compute_sobs(params, dataset)
        computed_params['sobs'] = sobs

        computed_params['alpha'] = params.get('alpha', 1.0)
        
        N = len(dataset)
        computed_params['lam_kl'] = params.get('lam_kl', 1.0/N)
        computed_params['lam_sl'] = params.get('lam_sl', 1.0/N)
        computed_params['tau'] = params.get('tau', 0.0)
        
        return computed_params

