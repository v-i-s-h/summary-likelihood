# Base class

import torch
from torch.utils.data.dataset import Dataset


class DatasetBase(Dataset):
    """
        Base class for all custom datasets
    """
    def __init__(self) -> None:
        super().__init__()
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = self.x[index]
        y = self.y[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y
