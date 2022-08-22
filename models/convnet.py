import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization

from methods.edl import compute_prob_from_evidence


class ConvNet(nn.Module):
    """
        
    """

    def __init__(self, K=10, 
            prior_mu=0.0, prior_sigma=np.exp(-1),
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()

        self.conv1 = Conv2dReparameterization(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            # stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = Conv2dReparameterization(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            # stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = Conv2dReparameterization(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = Conv2dReparameterization(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = Conv2dReparameterization(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        self.bn5 = nn.BatchNorm2d(num_features=64)

        self.fc1 = LinearReparameterization(
            in_features=64 * 4 * 4,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterization(
            in_features=128,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc3 = LinearReparameterization(
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
        
        x, kl = self.conv1(x)
        kl_sum += kl
        x = self.bn1(x)
        x = F.relu(x)

        x, kl = self.conv2(x)
        kl_sum += kl
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x, kl = self.conv3(x)
        kl_sum += kl
        x = self.bn3(x)
        x = F.relu(x)

        x, kl = self.conv4(x)
        kl_sum += kl
        x = self.bn4(x)
        x = F.relu(x)

        x, kl = self.conv5(x)
        kl_sum += kl
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)

        x, kl = self.fc2(x)
        kl_sum += kl
        x = F.relu(x)

        x, kl = self.fc3(x)
        kl_sum += kl

        return x, kl_sum

    def get_softmax(self, x):
        logits, _ = self.get_logits(x)
        scores = F.softmax(logits, dim=1)

        return scores


class ConvNetEDL(nn.Module):
    """
        
    """

    def __init__(self, K=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=3
        )
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn5 = nn.BatchNorm2d(num_features=64)

        self.fc1 = nn.Linear(
            in_features=64 * 4 * 4,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=128
        )

        self.fc3 = nn.Linear(
            in_features=128,
            out_features=K
        )

        self.num_classes = K

    def forward(self, x):
        evidence = self.get_evidence(x)

        return evidence

    def get_evidence(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)


        x = F.relu(x) # Evidence are ReLU'd
        
        return x

    def get_softmax(self, x):
        evidence = self.get_evidence(x)
        scores = compute_prob_from_evidence(evidence)

        return scores
    

if __name__ == "__main__":
    x = torch.rand(1, 1, 28, 28)

    m = ConvNet(K=2)

    y, kl = m(x)
    p = torch.exp(y)

    print(y, kl)
    print(p)
