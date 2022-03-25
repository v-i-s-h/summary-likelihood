"""
    Utility functions
"""

import json

import torch
import torch.nn as nn


class SoftHistogram(nn.Module):
    """
        Create soft histogram from samples
    """
    def __init__(self, bins, min, max, sigma):
        """
        Parameters
        ----------
        bins : int
            Number of bins in histogram
        min : float
            Minumum value
        max : float
            Maximum value
        sigma : float
            Slope of sigmoid
        """
        super().__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.50)
        self.centers = nn.Parameter(self.centers, requires_grad=False)

    def forward(self, x):
        """Computes soft histogram"""
        x = torch.unsqueeze(x, 1) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=-1) + 1e-6 # epsilon for zero bins
        x = x / x.sum(dim=-1).unsqueeze(1)

        return x


def parse_params_str(params_str):
    """
        parse_params_str
    Parses parameter string and returns a dictionary of parameters
    """
    buffer = []

    try:
        params = params_str.split(',')
        for p in params:
            try:
                name, value = p.split('=')
                buffer.append('"{}":{}'.format(name, value))
            except ValueError as err:
                # Not a key-value pair
                # Treta this is as enable flag
                buffer.append('"{}":true'.format(p.replace('-', '_')))
    except AttributeError as err:
        # No value passed
        pass
    # make JSON string
    buffer = '{' + ','.join(buffer) + '}'
    
    # Off load parsing to JSON
    params_dict = json.loads(buffer)
    
    return params_dict


if __name__ == "__main__":

    str1 = "a=1,b=2,flag=true,enable"
    r1 = parse_params_str(str1)
    assert r1['a'] == 1 and r1['b'] == 2 and r1['enable'] and r1['flag']
