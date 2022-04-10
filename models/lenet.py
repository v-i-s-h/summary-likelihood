import torch
import torch.nn as nn
import torch.nn.functional as F

from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization

class LeNet(nn.Module):
    def __init__(self, K=10, 
            prior_mu=0.0, prior_sigma=1.0,
            posterior_mu_init=0.0, posterior_rho_init=-3.0):
        super().__init__()
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = Conv2dReparameterization(
            in_channels=1,
            out_channels=6,
            kernel_size=5,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.conv2 = Conv2dReparameterization(
            in_channels=6,
            out_channels=16,
            kernel_size=5,
            stride=1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )
        
        self.fc1 = LinearReparameterization(
            in_features=16 * 4 * 4,
            out_features=120,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterization(
            in_features=120,
            out_features=84,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc3 = LinearReparameterization(
            in_features=84,
            out_features=K,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.num_classes = K

    def forward(self, x):
        kl_sum = 0.0

        x, kl = self.conv1(x)
        kl_sum += kl
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x, kl = self.conv2(x)
        kl_sum += kl
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
        
        output = F.log_softmax(x, dim=1)
        
        return output, kl_sum
