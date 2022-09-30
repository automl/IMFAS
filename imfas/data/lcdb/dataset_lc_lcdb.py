from pathlib import Path

import torch
from torch.utils.data import Dataset


class Dataset_LC_LCDB(Dataset):
    """Dataset_LC but for LCDB. Due to the sizeof the dataset, the tensor.pt file
    is lazily loaded.
    """

    def __init__(self, root, split, transform=None):
        self.root = Path(root)

        self.split = split
        self.files = split  # set(self.root.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        # load the tensor.pt file
        lc_tensor = torch.load(self.root / self.files[idx] + '.pt')

        if self.transform:
            lc_tensor = self.transform(lc_tensor)

        return lc_tensor
