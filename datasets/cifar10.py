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