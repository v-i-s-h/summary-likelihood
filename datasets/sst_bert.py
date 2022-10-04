# SST BERT Embeddings dataset
# Note: run data/create_sst_emb.py before using this dataset


import os
import pickle
import numpy as np

from .dataset import DatasetBase

class SSTBERT(DatasetBase):
    def __init__(self, split,
            corruption=None,
            transform=None,
            root=os.path.abspath(os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 
                    "./../data/"))) -> None:
        super().__init__()

        self.split = split
        self.root = root
        self.transform = transform

        if corruption:
            # Corruption will be given in the format eps-0.50
            _, gamma = corruption.split('-')
            gamma = float(gamma)
            self.gamma = gamma
        else:
            self.gamma = None

        # Load correct split data
        split_file = {
            'train': 'emb_train.pkl',
            'test': 'emb_test.pkl',
            'val': 'emb_dev.pkl'
        }.get(split)
        datafile = os.path.join(root, "SST", split_file)
        d = None
        with open(datafile, 'rb') as f:
            d = pickle.load(f)
            
        self.x = d['embeddings']
        self.y = np.array(d['labels'])

        if self.gamma:
            assert self.gamma >= 0.0 and self.gamma <=1, "gamma must be in [0, 1]"
            eps = np.random.randn(*self.x.shape).astype(np.float32)
            # Variance preserving noise addition
            self.x = (1.0 - self.gamma) * self.x + self.gamma * eps

        self.n_labels = 2
        self.n_samples = self.y.shape[0]
        self.n1 = np.sum(self.y) # because labels are 0-1
        self.n0 = self.n_samples - self.n1
        self.pos_weight = self.n0 / self.n1
        self.n_classes = [self.n0, self.n1]

    def __repr__(self):
        return "SST-BERT" + \
                "\n    Split                 : {}".format(self.split) + \
                "\n    X shape               : {}".format(self.x.shape) + \
                "\n    Y shape               : {}".format(self.y.shape) + \
                "\n    No.of positive classes: {} ({:.2f}%)".format(
                    self.n1, 100 * self.n1 / self.n_samples) + \
                "\n    gamma                 : {:5.3f}".format(
                    self.gamma if self.gamma else 0.0)


if __name__ == "__main__":
    
    ds_train = SSTBERT('train', corruption='eps-0.2')
    print(ds_train)

    ds_val = SSTBERT('val')
    print(ds_val)

    ds_test = SSTBERT('test')
    print(ds_test)
