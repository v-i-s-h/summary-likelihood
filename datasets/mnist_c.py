# Dataset for MNIST C dataset

from multiprocessing.sharedctypes import Value
import os
import numpy as np
from .dataset import DatasetBase


class BinaryMNISTC(DatasetBase):
    """
        Corrupted MNIST with only two labels
    """

    corruptions = [
        'brightness',
        'canny_edges',
        'dotted_line',
        'fog',
        'glass_blur',
        'identity',
        'impulse_noise',
        'motion_blur',
        'rotate',
        'scale',
        'shear',
        'shot_noise',
        'spatter',
        'stripe',
        'translate',
        'zigzag'
    ]

    def __init__(self,
            labels,
            corruption,
            split,
            imbalance=None,
            transform=None,
            size=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/"))):
        super().__init__()


        assert labels > 9 and labels < 100, "Sublabels can be only two digits"
        
        if corruption not in self.corruptions:
            raise ValueError("Unknown corruption '{}'. Possible corruptions are [{}]".format(
                corruption,
                ', '.join(self.corruptions)
            ))

        self.labels = [int(l) for l in str(labels)]
        self.corruption = corruption
        self.split = split
        self.transform = transform
        self.root = root
        self.pos_weight = 1.0
        self.imbalance = imbalance

        if imbalance:
            assert imbalance <= 0.50, "Imbalance should be less than 0.50!"

        if self.split == 'train':
            images_file = os.path.join(self.root, 'mnist_c', self.corruption, 'train_images.npy')
            labels_file = os.path.join(self.root, 'mnist_c', self.corruption, 'train_labels.npy')
        elif self.split == 'val':
            images_file = os.path.join(self.root, 'mnist_c', self.corruption, 'train_images.npy')
            labels_file = os.path.join(self.root, 'mnist_c', self.corruption, 'train_labels.npy')
        elif self.split == 'test':
            images_file = os.path.join(self.root, 'mnist_c', self.corruption, 'test_images.npy')
            labels_file = os.path.join(self.root, 'mnist_c', self.corruption, 'test_labels.npy')
        else:
            raise ValueError
        
        x = np.load(images_file)
        y = np.load(labels_file)

        # Filter unnecessary classes out
        idx0 = np.where(y == self.labels[0])[0]
        idx1 = np.where(y == self.labels[1])[0]
        if self.imbalance:
            n0 = len(idx0)
            n1 = len(idx1)
            n1_new = int( self.imbalance / (1.0 - self.imbalance) * n0) # Label 1 is minority class
            if n1_new < n1:
                idx1 = np.random.choice(idx1, size=n1_new, replace=False)
            else:
                print("WARNING: Not enough samples to get target imbalance.")

        idx = np.hstack((idx0, idx1))
        if size is not None:
            if idx.shape[0] > size:
                idx = np.random.choice(idx, size=size, replace=False)
            else:
                raise ValueError("Not enough samples ({}) for target dataset size ({})".format(
                    idx.shape[0], size
                ))
        idx = np.sort(idx) # index of selected samples
        
        self.x = np.float32(x[idx])
        self.y = y[idx]

        # Rescale x to 0 - 1 range
        self.x = self.x / 255.0

        # Convert labels to 0 and 1
        idx0 = np.where(self.y == self.labels[0])[0]
        idx1 = np.where(self.y == self.labels[1])[0]
        self.y[idx0] = 0
        self.y[idx1] = 1
        self.y = self.y

        # Set sample weight for minority class
        self.n_labels = 2 # Binary dataset
        self.n_samples = self.y.shape[0]
        self.n1 = idx1.shape[0]
        self.n0 = self.n_samples - self.n1
        self.pos_weight = self.n0 / self.n1
        self.n_classes = [self.n0, self.n1]

    def __repr__(self):
        return "BInaryMNIST-{} :: {}{}".format(self.corruption, *self.labels) + \
                "\n    Split                 : {}".format(self.split) + \
                "\n    X shape               : {}".format(self.x.shape) + \
                "\n    Y shape               : {}".format(self.y.shape) + \
                "\n    No.of positive classes: {} ({:.2f}%)".format(
                    self.n1, 100 * self.n1 / self.n_samples)