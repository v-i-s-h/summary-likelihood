# SVHN dataset

import os
import numpy as np
import torch
from torchvision import datasets
from .dataset import DatasetBase
from torch.utils.data import random_split
from numpy.random import Generator, PCG64


class SVHN(DatasetBase):
    """
        Wrapper for SVHN Pytorch dataset
    """

    def __init__(self, 
            split, 
            transform=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/SVHN/"))) -> None:
        super().__init__()

        self.ds = None
        self.split = split
        if split == 'test':
            # Load predefined test split
            _ds = datasets.SVHN(root, 
                    split='test', 
                    transform=transform,
                    download=True)
            n = len(_ds)
            self.ds, _ = random_split(_ds, [n, 0], # Make test also a SubSet for type consistency
                generator=torch.Generator().manual_seed(42)) # deterministic split
        elif split == 'train' or split == 'val':
            # Load train split and divide
            _ds = datasets.SVHN(root, 
                        split='train', 
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

    def __repr__(self):
        return "SVHN" + \
               "\n    Split        : {}".format(self.split) + \
               "\n    #Samples     : {}".format(len(self.ds))


def test_svhn():
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = SVHN('train', transform=transform)
    print(trainset)

    valset = SVHN('val', transform=transform)
    print(valset)

    testset = SVHN('test', transform=transform)
    print(testset)
    loader = torch.utils.data.DataLoader(testset, batch_size=10)
    images, labels = iter(loader).next()
    print("images = {}     labels = {}".format(images.shape, labels.shape))


if __name__=="__main__":
    test_svhn()