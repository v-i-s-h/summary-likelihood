# CIFAR-10 dataset

import os
import numpy as np
import torch
from torchvision import datasets
from .dataset import DatasetBase
from torch.utils.data import TensorDataset, random_split
from numpy.random import Generator, PCG64


class CIFAR10(DatasetBase):
    """
        Wrapper for CIFAR10 Pytorch dataset
    """
    corruptions = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'gaussian_blur',
        'gaussian_noise',
        'glass_blur',
        'identity',
        'impulse_noise',
        'jpeg_compression',
        'motion_blur',
        'pixelate',
        'saturate',
        'shot_noise',
        'snow',
        'spatter',
        'speckle_noise',
        'zoom_blur'
    ]

    def __init__(self, 
            split, 
            corruption=None, # Can be zoom_blur, or zoom_blur-5
            transform=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/"))) -> None:
        super().__init__()

        self.root = root
        self.ds = None
        self.x = None
        self.y = None
        self.transform = transform
        self.split = split

        if corruption:
            corruption_severity_pair = corruption.split('-')
            self.corruption = corruption_severity_pair[0]
            if len(corruption_severity_pair) > 1:
                self.severity_level = int(corruption_severity_pair[1])
            else:
                self.severity_level = 3 # Default level
        else:
            self.corruption = 'identity'
            self.severity_level = None

        # For no corruption or identity, use original CIFAR10
        if self.corruption == 'identity' or not self.corruption:
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
 
        else:
            # We have one of the corruptions
            if self.corruption not in self.corruptions:
                raise ValueError("Unknown corruption '{}'. Possible corruptions are [{}]".format(
                    corruption,
                    ', '.join(self.corruptions)
                ))

            # Only test split is defined for corruptions
            if split != 'test':
                raise ValueError("Only 'test' split is defined for corruptions.")

            images_file = os.path.join(self.root, 'CIFAR-10-C', '{}.npy'.format(
                            self.corruption))
            labels_file = os.path.join(self.root, 'CIFAR-10-C', 'labels.npy')

            x = np.load(images_file)
            y = np.load(labels_file)

            # Get the partition for severity level
            self.x = x[10000 * (self.severity_level - 1): 10000 * (self.severity_level - 1) + 10000]
            self.y = y[10000 * (self.severity_level - 1): 10000 * (self.severity_level - 1) + 10000]
            self.n_samples = self.y.shape[0]

        self.n_labels = 10

        # Custom len
        if self.ds:
            self.len_fn = self.ds.__len__
            self.getitem_fn = self.ds.__getitem__
        else:
            self.len_fn = super().__len__
            self.getitem_fn = super().__getitem__

    
    def __len__(self):
        # return self.ds.__len__()
        return self.len_fn()

    def __getitem__(self, index):
        # return self.ds.__getitem__(index)
        return self.getitem_fn(index)

    def __repr__(self):
        return "CIFAR10 :: {}".format(self.corruption) + \
               "\n    Split        : {}".format(self.split) + \
               "\n    Corruption   : {} - {}".format(self.corruption, self.severity_level) + \
               "\n    #Samples     : {}".format(len(self.ds) if self.ds else self.x.shape[0])


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


def test_cifar10im():
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


def test_cifar10():
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # trainset = CIFAR10('train', transform=transform)
    # print(trainset)

    # valset = CIFAR10('val', transform=transform)
    # print(valset)

    # testset = CIFAR10('test', transform=transform)
    # print(testset)
    # loader = torch.utils.data.DataLoader(testset, batch_size=10)
    # images, labels = iter(loader).next()
    # print("images = {}     labels = {}".format(images.shape, labels.shape))

    # Corruptions
    dataset = CIFAR10('test', corruption='identity', transform=transform)
    print(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    images, labels = iter(loader).next()
    print("images = {}     labels = {}".format(images.shape, labels.shape))
    # print(images[0, 0, :5, :], labels[0], type(labels[0]))

    # Corruptions - zoom_blur
    dataset = CIFAR10('test', corruption='zoom_blur', transform=transform)
    print(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=10)
    images, labels = iter(loader).next()
    print("images = {}     labels = {}".format(images.shape, labels.shape))
    # print(images[0, 0, :5, :], labels[0], type(labels[0]))


if __name__ == "__main__":
    
    # test_cifar10im()
    test_cifar10()
