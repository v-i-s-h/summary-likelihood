import torch
# Model based on: https://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from bayesian_torch.layers import Conv2dReparameterization
from bayesian_torch.layers import LinearReparameterization

from methods.edl import compute_prob_from_evidence


class CifarNetEDL(nn.Module):
    """
        
    """

    def __init__(self, K=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=20,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=20,
            out_channels=20,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        self.fc1 = nn.Linear(
            in_features=20 * 4 * 4,
            out_features=10
        )

        self.num_classes = K

        # Note: evidence prior is defined here, but will be injected into
        # the model by edl method
        self.evidence_prior = None

    def forward(self, x):
        evidence = self.get_evidence(x)

        return evidence

    def get_evidence(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x) 

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)  

        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension

        x = self.fc1(x)
        x = F.relu(x)

        # x = self.fc2(x)
        # x = F.relu(x)

        # x = self.fc3(x)


        # x = F.relu(x) # Evidence are ReLU'd
        
        return x

    def get_softmax(self, x):
        evidence = self.get_evidence(x)
        if self.evidence_prior is not None:
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)
        else:
            print('WARNING: Unknown evidence prior for EDL model. Using uniform evidence')
            self.evidence_prior = torch.ones(self.num_classes, device=x.device)
            scores = compute_prob_from_evidence(self.evidence_prior, evidence)

        return scores


if __name__ == "__main__":
    model = CifarNetEDL(10)

    print(model)

    x = torch.rand((5, 3, 32, 32))
    y = model(x)

    print(y.shape)
