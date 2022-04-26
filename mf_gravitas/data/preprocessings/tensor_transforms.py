import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

import pdb

from mf_gravitas.data.preprocessings.table_transforms import Transform



class ScaleStd(Transform):
    def __init__(self):
        """
        Warning: only dfs where all columns have the same type can be converted to tensor!
        --> i.e. categoricals must be converted to numerical features!
        """
        super(ScaleStd, self).__init__()

    def fit(self, X, dtype=torch.float32):
        
        y = X - X.mean()
        y = y / y.norm()

        return y