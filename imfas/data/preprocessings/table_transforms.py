from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class Transform(nn.Module):
    # FIXME: nn.Module inheritance is required by nn.Sequential
    """
    For table based data.

    Piggy-bagging sklearn's and torchvision's pipeline classes
    """

    def __init__(self, columns: Tuple[str] = None):
        super().__init__()
        self.columns = columns

    def fit(self, X: pd.DataFrame):
        # get all the necessary statistics, that will be used in both the train and
        # test set - to ensure that the same scales are used for the transforms.
        return self

    def transform(self, X: pd.DataFrame):
        # once the transformer is fit,
        # this method can be called to actually transform
        return X


class Scalar(StandardScaler, Transform):
    def __init__(self, columns: list, copy=True, with_mean=True, with_std=True):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns

    def fit(self, X: pd.DataFrame):
        StandardScaler.fit(self, X[self.columns])
        return self

    def transform(self, X: pd.DataFrame):
        X[self.columns] = StandardScaler.transform(self, X[self.columns])
        return X


class ToTensor(Transform):
    def __init__(self, columns=None):
        """
        Warning: only dfs where all columns have the same type can be converted to tensor!
        --> i.e. categoricals must be converted to numerical features!
        """
        super(ToTensor, self).__init__()
        self.columns = columns

    def transform(self, X: pd.DataFrame, dtype=torch.float32):
        # print(X.values)
        # pdb.set_trace()

        return torch.tensor(X.values, dtype=dtype)


class Nan_zero(Transform):
    def __init__(self):
        """
        transform a,ll nan values to 0
        """
        super(Nan_zero, self).__init__()

    def transform(self, X: pd.DataFrame):
        X.fillna(0.0, inplace=True)

        return X


class Nan_mean(Transform):
    def __init__(self):
        """
        transform a,ll nan values to 0
        """
        super(Nan_zero, self).__init__()

    def transform(self, X: pd.DataFrame):
        X.fillna(0.0, inplace=True)

        return X


class Drop(Transform):
    def __init__(self, columns):
        """
        Warning: only dfs where all columns have the same type can be converted to tensor!
        --> i.e. categoricals must be converted to numerical features!
        """

        self.columns = columns

    def transform(self, X: pd.DataFrame):
        return X.drop(columns=self.columns)


class Convert(Transform):
    def __init__(self, columns=None, dtype=None):  # FIXME: dtype should be used
        """change to a specific type"""

        self.columns = columns
        self.dtype = dtype

    def transform(self, X: pd.DataFrame):
        if self.columns is None:
            self.columns = X.columns

        for col in self.columns:
            if self.dtype == "int":
                X[col] = X[col].astype(int)
            else:
                X[col] = pd.to_numeric(X[col])
        return X


class Replace(Transform):
    def __init__(self, columns, replacedict):
        """wrapper to pd.DataFrame.replace"""

        self.columns = columns
        self.replacedict = replacedict

    def transform(self, X: pd.DataFrame):
        for col in self.columns:
            X[col].replace(self.replacedict, inplace=True)
        return X
