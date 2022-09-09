# Fashion MNIST dataset

import os
import torch
from torchvision import datasets
from .dataset import DatasetBase
from torch.utils.data import random_split
from numpy.random import Generator, PCG64


class FMNIST(DatasetBase):
    """
        Wrapper for FashionMNIST Pytorch dataset
    """

    def __init__(self,
            split,
            transform=None,
            root=os.path.abspath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "./../data/"))) -> None:
        super().__init__()

        self.ds = None
        if split == 'test':
            _ds = datasets.FashionMNIST(root,
                    train = False,
                    transform=transform,
                    download=True)
            n = len(_ds)
            self.ds, _ = random_split(_ds, [n, 0], # Make test also a SubSet for type consistency
                generator=torch.Generator().manual_seed(42)) # deterministic split
        elif split == 'train' or split == 'val':
            _ds = datasets.FashionMNIST(root,
                    train=True,
                    transform=transform,
                    download=True)
            n = len(_ds)
            n_val = int(n * 0.10 + 0.5) # Use 10% as validation set

            _ds_train, _ds_val = random_split(_ds, [n - n_val, n_val],
                generator=torch.Generator().manual_seed(42)) # deterministic split

            if split == 'train':
                self.ds = _ds_train
            else:
                self.ds = _ds_val
        else:
            raise KeyError("Unknown data split '{}'".format(split))

        self.n_labels = 10

    def __len__(self):
        return self.ds.__len__()

    def __getitem__(self, index):
        return self.ds.__getitem__(index)
