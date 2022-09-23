"""
Source: https://nn.labml.ai/uncertainty/evidence/index.html
"""

import os
import argparse
import json
from datetime import datetime
from typing import Any
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from torch.utils.data import DataLoader
import torchmetrics

from datasets import CIFAR10
from transforms import normalize_x_cifar

from tqdm import tqdm

# ------------------------------------------------------------------------------------------

class Schedule:
    def __call__(self, x):
        raise NotImplementedError()


class Piecewise(Schedule):
    """
    ## Piecewise schedule
    """

    def __init__(self, endpoints: List[Tuple[float, float]], outside_value: float = None):
        """
        ### Initialize
        `endpoints` is list of pairs `(x, y)`.
         The values between endpoints are linearly interpolated.
        `y` values outside the range covered by `x` are
        `outside_value`.
        """

        # `(x, y)` pairs should be sorted
        indexes = [e[0] for e in endpoints]
        assert indexes == sorted(indexes)

        self._outside_value = outside_value
        self._endpoints = endpoints

    def __call__(self, x):
        """
        ### Find `y` for given `x`
        """

        # iterate through each segment
        for (x1, y1), (x2, y2) in zip(self._endpoints[:-1], self._endpoints[1:]):
            # interpolate if `x` is within the segment
            if x1 <= x < x2:
                dx = float(x - x1) / (x2 - x1)
                return y1 + dx * (y2 - y1)

        # return outside value otherwise
        return self._outside_value

    def __str__(self):
        endpoints = ", ".join([f"({e[0]}, {e[1]})" for e in self._endpoints])
        return f"Schedule[{endpoints}, {self._outside_value}]"


class RelativePiecewise(Piecewise):
    def __init__(self, relative_endpoits: List[Tuple[float, float]], total_steps: int):
        endpoints = []
        for e in relative_endpoits:
            index = int(total_steps * e[0])
            assert index >= 0
            endpoints.append((index, e[1]))

        super().__init__(endpoints, outside_value=relative_endpoits[-1][1])

# ------------------------------------------------------------------------------------------

class SquaredErrorBayesRisk(nn.Module):
    """
    <a id="SquaredErrorBayesRisk"></a>
    ## Bayes Risk with Squared Error Loss
    Here the cost function is squared error,
    $$\sum_{k=1}^K (y_k - p_k)^2 = \Vert \mathbf{y} - \mathbf{p} \Vert_2^2$$
    We integrate this cost over all $\mathbf{p}$
    \begin{align}
    \mathcal{L}(\Theta)
    &= -\log \Bigg(
     \int
      \Big[ \sum_{k=1}^K (y_k - p_k)^2 \Big]
      \frac{1}{B(\textcolor{orange}{\mathbf{\alpha}})}
      \prod_{k=1}^K p_k^{\textcolor{orange}{\alpha_k} - 1}
     d\mathbf{p}
     \Bigg ) \\
    &= \sum_{k=1}^K \mathbb{E} \Big[ y_k^2 -2 y_k p_k + p_k^2 \Big] \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big)
    \end{align}
    Where $$\mathbb{E}[p_k] = \hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$$
    is the expected probability when sampled from the Dirichlet distribution
    and $$\mathbb{E}[p_k^2] = \mathbb{E}[p_k]^2 + \text{Var}(p_k)$$
     where
    $$\text{Var}(p_k) = \frac{\textcolor{orange}{\alpha_k}(S - \textcolor{orange}{\alpha_k})}{S^2 (S + 1)}
    = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$$
     is the variance.
    This gives,
    \begin{align}
    \mathcal{L}(\Theta)
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] + \mathbb{E}[p_k^2] \Big) \\
    &= \sum_{k=1}^K \Big( y_k^2 -2 y_k \mathbb{E}[p_k] +  \mathbb{E}[p_k]^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( \big( y_k -\mathbb{E}[p_k] \big)^2 + \text{Var}(p_k) \Big) \\
    &= \sum_{k=1}^K \Big( ( y_k -\hat{p}_k)^2 + \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1} \Big)
    \end{align}
    This first part of the equation $\big(y_k -\mathbb{E}[p_k]\big)^2$ is the error term and
    the second part is the variance.
    """

    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # $S = \sum_{k=1}^K \textcolor{orange}{\alpha_k}$
        strength = alpha.sum(dim=-1)
        # $\hat{p}_k = \frac{\textcolor{orange}{\alpha_k}}{S}$
        p = alpha / strength[:, None]

        # Error $(y_k -\hat{p}_k)^2$
        err = (target - p) ** 2
        # Variance $\text{Var}(p_k) = \frac{\hat{p}_k(1 - \hat{p}_k)}{S + 1}$
        var = p * (1 - p) / (strength[:, None] + 1)

        # Sum of them
        loss = (err + var).sum(dim=-1)

        # Mean loss over the batch
        return loss.mean()

class KLDivergenceLoss(nn.Module):
    """
    <a id="KLDivergenceLoss"></a>
    ## KL Divergence Regularization Loss
    This tries to shrink the total evidence to zero if the sample cannot be correctly classified.
    First we calculate $\tilde{\alpha}_k = y_k + (1 - y_k) \textcolor{orange}{\alpha_k}$ the
    Dirichlet parameters after remove the correct evidence.
    \begin{align}
    &KL \Big[ D(\mathbf{p} \vert \mathbf{\tilde{\alpha}}) \Big \Vert
    D(\mathbf{p} \vert <1, \dots, 1>\Big] \\
    &= \log \Bigg( \frac{\Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)}
    {\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \Bigg)
    + \sum_{k=1}^K (\tilde{\alpha}_k - 1)
    \Big[ \psi(\tilde{\alpha}_k) - \psi(\tilde{S}) \Big]
    \end{align}
    where $\Gamma(\cdot)$ is the gamma function,
    $\psi(\cdot)$ is the $digamma$ function and
    $\tilde{S} = \sum_{k=1}^K \tilde{\alpha}_k$
    """
    def forward(self, evidence: torch.Tensor, target: torch.Tensor):
        """
        * `evidence` is $\mathbf{e} \ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        # $\textcolor{orange}{\alpha_k} = e_k + 1$
        alpha = evidence + 1.
        # Number of classes
        n_classes = evidence.shape[-1]
        # Remove non-misleading evidence
        # $$\tilde{\alpha}_k = y_k + (1 - y_k) \textcolor{orange}{\alpha_k}$$
        alpha_tilde = target + (1 - target) * alpha
        # $\tilde{S} = \sum_{k=1}^K \tilde{\alpha}_k$
        strength_tilde = alpha_tilde.sum(dim=-1)

        # The first term
        #
        # \begin{align}
        # &\log \Bigg( \frac{\Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)}
        #     {\Gamma(K) \prod_{k=1}^K \Gamma(\tilde{\alpha}_k)} \Bigg) \\
        # &= \log \Gamma \Big( \sum_{k=1}^K \tilde{\alpha}_k \Big)
        #   - \log \Gamma(K)
        #   - \sum_{k=1}^K \log \Gamma(\tilde{\alpha}_k)
        # \end{align}
        first = (torch.lgamma(alpha_tilde.sum(dim=-1))
                 - torch.lgamma(alpha_tilde.new_tensor(float(n_classes)))
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))

        # The second term
        # $$\sum_{k=1}^K (\tilde{\alpha}_k - 1)
        #     \Big[ \psi(\tilde{\alpha}_k) - \psi(\tilde{S}) \Big]$$
        second = (
                (alpha_tilde - 1) *
                (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde)[:, None])
        ).sum(dim=-1)

        # Sum of the terms
        loss = first + second

        # Mean loss over the batch
        return loss.mean()

# ------------------------------------------------------------------------------------------

from models import VGG11EDL

def main():
    # Config
    batch_size = 256
    n_epochs = 500

    # Timestamp for experiment
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Set up command line args parser
    parser = argparse.ArgumentParser()
    # Others
    parser.add_argument('--outdir', type=str, required=False, default="zoo/test/",
            help="Parent output directory to save model")
    parser.add_argument('--prefix', type=str, required=False, default=None,
            help="Prefix for model directory")
    parser.add_argument('--seed', type=int, required=False, default=None,
            help="Seed for running experiment")
    args = parser.parse_args()

    seed = args.seed
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            args.outdir, 
                            'CIFAR10', 
                            'VGG11EDL', 
                            args.prefix+"-"+timestamp if args.prefix else timestamp)
    os.makedirs(outdir, exist_ok=True)

    # Writeout experiment configuration
    config_file_path = os.path.join(outdir, "config.json")
    with open(config_file_path, 'w') as fp:
        json.dump({
            'method': 'edl',
            'method_params': {},
            'dataset': 'CIFAR10',
            'ds_params': {},
            'transform': 'normalize_x_cifar',
            'model': 'VGG11EDL',
            'model_params': {},
            'max_steps': 0,
            'batch_size': batch_size,
            'wt_loss': False,
            'mc_samples': 32,
            'seed': seed
        }, fp, indent=2)

    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("INFO: Using seed `{}`.".format(seed))

    print("Training EDL")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = VGG11EDL(10).to(device=device)

    

    x_transform = normalize_x_cifar()
    trainset = CIFAR10('train', transform=x_transform)
    tr_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    valset = CIFAR10('val', transform=x_transform)
    val_loader = DataLoader(dataset=valset, batch_size=8*batch_size, shuffle=False)
    testset = CIFAR10('test', transform=x_transform)
    test_loader = DataLoader(dataset=testset, batch_size=8*batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
    kl_div_loss_func = KLDivergenceLoss()
    loss_func = SquaredErrorBayesRisk()
    n_train = len(trainset)
    kl_div_coef = RelativePiecewise(
                    [(0, 0.), (0.2, 0.01), (1, 1.)],
                    n_epochs * n_train)

    val_acc = torchmetrics.Accuracy().to(device)

    model.train()
    optimizer.zero_grad()
    val_acc.reset()
    last_best_val_acc = 0.0
    for epoch_idx in tqdm(range(n_epochs), desc="Training"):
        for batch_idx, (x, y) in enumerate(tr_loader):

            x = x.to(device)
            y = y.to(device)
            eye = torch.eye(10).to(torch.float).to(device)
            y = eye[y]

            evidence = model(x)

            loss = loss_func(evidence, y)
            kl_div_loss = kl_div_loss_func(evidence, y)

            annealing_coef = 0.001 * min(1., 
                                kl_div_coef(
                                    n_train * epoch_idx + batch_idx * batch_size
                            ))

            total_loss = loss + annealing_coef * kl_div_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # print(total_loss.item())

        # Run validation accuracy
        for batch_idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)
            
            evidence = model(x)

            # Highest evidence for predictec class
            preds = evidence
            acc = val_acc(preds, y)
        acc = val_acc.compute()
        print("validation accuracy = ", acc, annealing_coef)
        val_acc.reset()

        print(last_best_val_acc, acc.item())
        if last_best_val_acc < acc.item():
            # Save this model
            torch.save(model.state_dict(), 
                        os.path.join(outdir, 'step=0000.ckpt'))
            print("saving model checkpoint")
            last_best_val_acc = acc.item()

    # Run test accuracy
    for batch_idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)
        
        evidence = model(x)

        # Highest evidence for predictec class
        preds = evidence
        acc = val_acc(preds, y)
    acc = val_acc.compute()
    print("Test accuracy = ", acc, annealing_coef)
    val_acc.reset()

    # Save model


        




#
if __name__ == '__main__':
    main()
