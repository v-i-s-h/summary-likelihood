"""
    Utility functions
"""

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



def convert(value):
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass
    return value

def parse_params_str(params_str):
    """
        parse_params_str
    Parses parameter string and returns a dictionary of parameters
    """
    params_dict = {}

    try:
        params = params_str.split(',')
        for p in params:
            try:
                name, value = p.split('=')
                # try to convert to int of float
                value = convert(value)
                params_dict[name] = value
            except ValueError as err:
                # Not a key-value pair
                # Treat this is as enable flag
                if p:
                    name = p.replace('-', '_')
                    params_dict[name] = True
    except AttributeError as err:
        # No value passed
        pass
    
    return params_dict


if __name__ == "__main__":

    str1 = "a=1,b=2,flag=true,enable"
    r1 = parse_params_str(str1)
    assert r1['a'] == 1 and isinstance(r1['a'], int) \
        and r1['b'] == 2 and isinstance(r1['b'], int) \
        and r1['enable'] \
        and r1['flag']

    str2 = "sublabel=53,corruption=identity,imbalance=0.10"
    r2 = parse_params_str(str2)
    assert r2['sublabel'] == 53 and isinstance(r2['sublabel'], int) \
        and r2['corruption'] == 'identity' and isinstance(r2['corruption'], str) \
        and r2['imbalance'] == 0.10 and isinstance(r2['imbalance'], float)

    str3 = ""
    r3 = parse_params_str(str3)
    assert not r3 # should be empty