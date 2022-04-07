# ChestMNISt dataset

import os
import numpy as np
from .dataset import DatasetBase




"""
    ChestMNIST dataset
"""
class ChestMNIST(DatasetBase):
    corruptions = [
        'identity'
    ]

    labels_dict = {
        "atelectasis": 0,
        "cardiomegaly": 1,
        "effusion": 2,
        "infiltration": 3,
        "mass": 4,
        "nodule": 5,
        "pneumonia": 6,
        "pneumothorax": 7,
        "consolidation": 8,
        "edema": 9,
        "emphysema": 10,
        "fibrosis": 11,
        "pleural": 12,
        "hernia": 13
    }


    def __init__(self,
            label,
            split,
            transform=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/"))):
        super().__init__()

        self.label = label
        if self.label is not None:
            self.label_idx = self.labels_dict[self.label]
        else:
            self.label_idx = None
        self.split = split
        self.transform = transform
        self.root = root
        self.pos_weight = 1.0
        
        npz_file = np.load(os.path.join(self.root, "chestmnist.npz"))

        if self.split == 'train':
            self.x = npz_file['train_images']
            if self.label_idx is not None:
                self.y = npz_file['train_labels'][:, self.label_idx].astype(float)
            else:
                self.y = npz_file['train_labels'].astype(float)
        elif self.split == 'val':
            self.x = npz_file['val_images']
            if self.label_idx is not None:
                self.y = npz_file['val_labels'][:, self.label_idx].astype(float)
            else:
                self.y = npz_file['val_labels'].astype(float)
        elif self.split == 'test':
            self.x = npz_file['test_images']
            if self.label_idx is not None:
                self.y = npz_file['test_labels'][:, self.label_idx].astype(float)
            else:
                self.y = npz_file['test_labels'].astype(float)
        else:
            raise ValueError

        # Set sample weight for minority class
        self.n_labels = 2
        self.n_samples = self.y.shape[0]
        self.n1 = self.y.sum().astype(int)
        self.n0 = self.n_samples - self.n1
        self.pos_weight = self.n0 / self.n1
        self.n_classes = [self.n0, self.n1]

    def __repr__(self):
        return "ChestMNIST :: {}".format(self.label) + \
                "\n    Split                 : {}".format(self.split) + \
                "\n    X shape               : {}".format(self.x.shape) + \
                "\n    Y shape               : {}".format(self.y.shape) + \
                "\n    No.of minority classes: {} ({:.2f}%)".format(
                    self.n1, 100 * self.n0 / self.n_samples)
