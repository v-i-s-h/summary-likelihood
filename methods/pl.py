from .base import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.stats as st
from scipy.special import betainc
from scipy.optimize import minimize


def compute_beta_prior_params(p0, ea):
    """
        Compute prior parameter for Beta base measure from proportion
        of majority class (0) and expected accuracy

    Parameters
    ----------
    p0 : float
        Proportion of majority class (class 0)
        0 > p0 > 1
    ea : float
        Expected accuracy 
        0 > ea >= 1

    Returns
    -------
    (a, b) for Beta distribution
    """

    # build objective function
    def objective(params, p0=p0, ea=ea):
        a = params[0]
        b = params[1]

        # Compute p0 with current params
        _p0 = st.beta(a, b).cdf(0.50)
        # Compute expected acc with current params
        _ea = _p0 - (a / (a+b)) * (2 * betainc(a+1, b, 0.50) - 1.0)

        # Compute error: simple l2 err is used
        err = (ea - _ea)**2 + (p0 - _p0)**2

        return err

    # Start with uniform
    a = 1.0
    b = 1.0
    # Minimize error
    result = minimize(objective, [a, b], 
                        # method='Nelder-Mead',
                        method='L-BFGS-B',
                        bounds=[(0.025, None), (0.025, None)], tol=1e-3)

    if result.success:
        a = result.x[0]
        b = result.x[1]
    else:
        a, b = None, None

    return a, b


def compute_sobs(params, dataset):
    # For s_obs
    base_measure = None

    if 'auto' in params:
        p0 = dataset.n0 / (dataset.n0 + dataset.n1)
        ea = params.get('ea', p0) # If no Ea, use p0 itself as Ea!

        a, b = compute_beta_prior_params(p0, ea)
        if a is not None and b is not None:
            _p0 = st.beta(a, b).cdf(0.50)
            _ea = _p0 - (a / (a+b)) * (2 * betainc(a+1, b, 0.50) - 1.0)
            print("INFO: Parameters for beta base measure : a = {:.4f}, b = {:.4f}".format(a, b))
            print("INFO: p0 = {:.3f}".format(_p0))
            print("INFO: Ea = {:.3f}".format(_ea))
        else:
            raise ValueError("Auto tuning prior: failed to find prior parameters.")

        base_measure = torch.distributions.Beta(torch.tensor(a), torch.tensor(b))
    elif 'beta' in params:
        base_measure = torch.distributions.Beta(torch.tensor(params['a']), torch.tensor(params['b']))
    else:
        raise ValueError("No information for base information given. Use 'auto' or 'beta'")

    return base_measure # We compute the likelihood of predictions w.r.t to this measure


class PredictionLikelihood(BaseModel):
    def __init__(self, model,
            sobs, alpha=1.0,
            lam_kl=1.0, lam_sl=1.0, tau=0.0,
            class_weight=None, mc_samples=32) -> None:
        # Checks
        assert tau >= 1e-3 or tau == 0.0, "tau needs to be greater than 1e-3 (or 0.0 to disable annealing"

        super().__init__(model, class_weight, mc_samples)

        self.lam_kl = lam_kl
        self.lam_sl = lam_sl
        
        self.base_measure = sobs
        self.alpha = alpha
        self.tau = tau # annealing parameter

        # Demo code: will work only for Binary classification
        if self.model.num_classes != 2:
            raise RuntimeError("PredictionLikelihood is implemented only for binary classification")

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
        if self.model.num_classes == 2:
            mc_y_pred = torch.stack([_y[:, 1] for _y in mc_y_pred])
        else:
            raise RuntimeError("PredictionLikelihood not implemented for multi class")
        yscore_samples = torch.exp(mc_y_pred) # y_pred are log_soft of label 1
        
        # log likelihood of predictions wrt to sobs
        try:
            ll_s_obs = self.base_measure.log_prob(yscore_samples)
        except Exception as e:
            print("************************")
            print(y_pred)
            print(mc_y_pred)
            print(yscore_samples)
            print("************************")
            raise e
        # raise RuntimeError("---------------------------------")

        
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
        self.log('dl_loss', sl_loss.detach())
        self.log('dl_annealed_scale', annlealed_scale)
        self.log('scaled_sl_loss', scaled_sl_loss.detach())

        return loss, y_pred

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
        if dataset.n_labels == 2:
            sobs = compute_sobs(params, dataset)
        else:
            raise RuntimeError("PreditionLikelihood is implemented only for Binary Classification")
        computed_params['sobs'] = sobs

        computed_params['alpha'] = params.get('alpha', 1.0)
        
        N = len(dataset)
        computed_params['lam_kl'] = params.get('lam_kl', 1.0/N)
        computed_params['lam_sl'] = params.get('lam_sl', 1.0/N)
        computed_params['tau'] = params.get('tau', 0.0)
        
        return computed_params

