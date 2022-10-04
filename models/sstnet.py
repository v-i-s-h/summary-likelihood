# SST binary classifier net

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import LinearReparameterization

from methods.edl import compute_prob_from_evidence


class SSTNet(nn.Module):
    def __init__(self, K=2, in_dim=768,
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        
        self.fc1 = LinearReparameterization(
            in_features=in_dim,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterization(
            in_features=128,
            out_features=K,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.num_classes = K

    def forward(self, x):
        
        logits, kl_sum = self.get_logits(x)
        output = F.log_softmax(logits, dim=1)
        
        return output, kl_sum
    
    def get_logits(self, x):
        kl_sum = 0.0

        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)

        x, kl = self.fc2(x)
        kl_sum += kl

        logits = x

        return logits, kl_sum

    def get_softmax(self, x):
        logits, _ = self.get_logits(x)
        scores = F.softmax(logits, dim=1)

        return scores


class SSTNetEDL(nn.Module):
    """
        Deterministic LeNet for EDL
        Outputs are ReLU-ed
    """
    def __init__(self, K=10, in_dim=768):
        super().__init__()
        
        self.fc1 = nn.Linear(
            in_features=768,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=K
        )

        self.num_classes = K

        # Note: evidence prior is defined here, but will be injected into
        # the model by edl method
        self.evidence_prior = None

    def forward(self, x):
        out = self.get_evidence(x)
        
        return out
    
    def get_evidence(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        evidence = F.relu(x)
        
        return evidence

    def get_softmax(self, x):
        evidence = self.get_evidence(x)
        if self.evidence_prior is not None:
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)
        else:
            print('WARNING: Unknown evidence prior for EDL model. Using uniform evidence')
            self.evidence_prior = torch.ones(self.num_classes, device=x.device)
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)

        return scores
