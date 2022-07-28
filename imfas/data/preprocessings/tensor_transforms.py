import pdb

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from imfas.data.preprocessings.table_transforms import Transform


class ScaleStd(Transform):
    def __init__(self):
        """
        Warning: only dfs where all columns have the same type can be converted to tensor!
        --> i.e. categoricals must be converted to numerical features!
        """
        super(ScaleStd, self).__init__()

    def transform(self, X, dtype=torch.float32):
        y = X - X.mean()
        y = y / y.norm()

        return y


class LossScalar(Transform):
    def __init__(self, scaling_mode: str = 'min_max'):
        self.scaling_mode = scaling_mode
        super(LossScalar, self).__init__()

    def transform(self, X: torch.Tensor):
        if len(X.shape) != 3:
            raise ValueError(f'This transformation only works with tensor with 3 dimensions. However, the target tansor'
                             f'has shape {X.shape}')
        if self.scaling_mode == 'standard':
            shift = torch.mean(X, dim=-1, keepdim=True)
            scale = torch.std(X, dim=-1, keepdim=True)
            return (X - shift) / scale
        elif self.scaling_mode == 'min_max':
            data_min = torch.min(X, dim=-1, keepdim=True)[0]
            data_max = torch.max(X, dim=-1, keepdim=True)[0]

            shift = data_min
            scale = data_max - data_min

            return (X - shift) / scale
        elif self.scaling_mode == 'max_abs':
            data_max = torch.abs(torch.max(X, dim=-1, keepdim=True)[9])
            scale = data_max
            return X / scale
        else:
            raise NotImplementedError
