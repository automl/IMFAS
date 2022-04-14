import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class Transform(nn.Module):
    # Fixme: nn.Module inheritance is required by nn.Sequential
    """
    For table based data.

    Piggy-bagging sklearn's and torchvision's pipeline classes
    """

    def __init__(self, columns: tuple[str] = None):
        super().__init__()
        self.columns = columns

    def fit(self, X):
        # get all the necessary statistics, that will be used in both the train and
        # test set - to ensure that the same scales are used for the transforms.
        return

    def transform(self, X):
        # once the transformer is fit,
        # this method can be called to actually transform
        return self.fit(X)


class Scalar(StandardScaler, Transform):

    def __init__(self, columns: list, copy=True, with_mean=True, with_std=True):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns

    def fit(self, X):
        StandardScaler.fit(self, X[self.columns])
        X[self.columns] = StandardScaler.transform(self, X[self.columns])
        return X

    def transform(self, X):
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

    def fit(self, X, dtype=torch.float32):
        return torch.tensor(X.values, dtype=dtype)


class Drop(Transform):
    def __init__(self, columns):
        """
        Warning: only dfs where all columns have the same type can be converted to tensor!
        --> i.e. categoricals must be converted to numerical features!
        """

        self.columns = columns

    def fit(self, X):
        return X.drop(columns=self.columns)


class Convert(Transform):
    def __init__(self, columns, dtype):
        """change to a specific type"""

        self.columns = columns
        self.dtype = dtype

    def fit(self, X):
        for col in self.columns:
            X[col] = pd.to_numeric(X[col])
        return X


class Replace(Transform):
    def __init__(self, columns, replacedict):
        """wrapper to pd.DataFrame.replace"""

        self.columns = columns
        self.replacedict = replacedict

    def fit(self, X):
        for col in self.columns:
            X[col].replace(self.replacedict, inplace=True)
        return X
