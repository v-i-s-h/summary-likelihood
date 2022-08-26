# CIFAR-10 dataset

import os
import numpy as np
import torch
from torchvision import datasets
from .dataset import DatasetBase
from torch.utils.data import random_split
from numpy.random import Generator, PCG64


class CIFAR10(DatasetBase):
    """
        Wrapper for CIFAR10 Pytorch dataset
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
            # Load predefined test split
            _ds = datasets.CIFAR10(root, 
                    train = False, 
                    transform=transform,
                    download=True)
            n = len(_ds)
            self.ds, _ = random_split(_ds, [n, 0], # Make test also a SubSet for type consistency
                generator=torch.Generator().manual_seed(42)) # deterministic split
        elif split == 'train' or split == 'val':
            # Load train split and divide
            _ds = datasets.CIFAR10(root, 
                        train = True, 
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


class CIFAR10Im(CIFAR10):
    def __init__(self, 
            split, 
            transform=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/"))) -> None:
        super().__init__(split=split, transform=transform, root=root)

        # Modify the sample to make class imbalance
        K = 10 # number of classes
        idxs = [] # list of indices of each class
        for i in range(K):
            idx_i = [j for j in self.ds.indices if self.ds.dataset.targets[j] == i]
            idxs.append(idx_i)
        
        
        # Deterministically select sample indices for each class
        for i in range(K):
            ni = int(np.round(len(idxs[i]) * 2 ** (-i)))
            if i == K-1:
                ni *= 2 # Only for last label, keep double the samples.
            idxs[i] = idxs[i][:ni] # Select first `ni` indices

        # Flatten into one list
        sample_ids = []
        for _idxs in idxs:
            sample_ids.extend(_idxs)
        # Shuffle 'deterministically' to break order
        Generator(PCG64(42)).shuffle(sample_ids)
        
        self.ds.indices = sample_ids


if __name__ == "__main__":
    import torch
    from torchvision import transforms
    CIFAR10Test = CIFAR10Im

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = CIFAR10Test(split='train', transform=transform)
    print("#sample => ", len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=12)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print("X::", images.shape)
    print("Y::", labels.shape, labels)
    print("loader len = ", len(trainloader))


    valset = CIFAR10Test(split='val', transform=transform)
    print("#sample => ", len(valset))
    loader = torch.utils.data.DataLoader(valset, batch_size=12)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape)
    print(labels.shape, labels)
    print("loader len = ", len(loader))


    testset = CIFAR10Test(split='test', transform=transform)
    print("#sample => ", len(testset))
    loader = torch.utils.data.DataLoader(testset, batch_size=12)
    dataiter = iter(loader)
    images, labels = dataiter.next()
    print(images.shape)
    print(labels.shape, labels)
    print("loader len = ", len(loader))
