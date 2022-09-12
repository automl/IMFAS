import pdb

import pandas as pd

from imfas.data.preprocessings.table_transforms import Transform


class Zero_fill(Transform):
    def __init__(self):
        """
        transform a,ll nan values to 0
        """
        super(Zero_fill, self).__init__()

    def transform(self, X: pd.DataFrame):
        X.fillna(0.0, inplace=True)

        return X


class Df_Mean(Transform):
    def __init__(self):
        """
        transform a,ll nan values to 0
        """
        super(Df_Mean, self).__init__()

    def transform(self, X: pd.DataFrame):
        X.fillna(X.mean(), inplace=True)

        return X


class Column_Mean(Transform):
    def __init__(self):
        """
        transform all nan values in each column to the mean value of thec column
        """
        super(Column_Mean, self).__init__()

    def transform(self, X: pd.DataFrame):

        for col in X.columns:
            X[col].fillna(X[col].mean(), inplace=True)

        return X


class Column_Ffill(Transform):
    def transform(self, X: pd.DataFrame):
        X_t = X.T.fillna(method="ffill", inplace=True)
        X_t = X_t.fillna(method="bfill", inplace=True)
        return X_t.T
