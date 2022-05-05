# CIFAR-10 dataset

import os
import numpy as np
from torchvision import datasets
from .dataset import DatasetBase


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

        self.ds = datasets.CIFAR10(root, 
                    train = split == 'train', 
                    transform=transform,
                    download=True)

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
            idx_i = [j for j in range(len(self.ds.targets)) if self.ds.targets[j] == i]
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
        # Shuffle to break order
        np.random.shuffle(sample_ids)
        
        self.ds.data = self.ds.data[sample_ids]
        self.ds.targets = [self.ds.targets[i] for i in sample_ids]


if __name__ == "__main__":
    import torch
    from torchvision import transforms
    from datasets import CIFAR10 as CIFAR10Test

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = CIFAR10Test(split='train', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=12)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    print(images.shape)
    print(labels.shape)