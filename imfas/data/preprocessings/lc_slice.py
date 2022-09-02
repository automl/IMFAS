from typing import List

import numpy as np
import pandas as pd
import torch

from imfas.data.preprocessings.table_transforms import Transform


# Since LC datasets are inherently pandas multi-index dataframes,
# a short indexing tutorial can be found here:
# https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe


class LC_TimeSlices(Transform):
    def __init__(self, slices: List[int]):
        """
        :param slices: which time slices across algos & dataset you want to have.
        """
        super().__init__(None)
        self.slices = slices

    def transform(self, X: pd.DataFrame):
        """
        :param X: is an Array, where the time dimension is #FIXME ???
        :return: torch.Tensor: (n_slices, n_datasets n_algorithms)
        # FIXME: downstream must be tensor!
        """
        # X[self.slices]
        format = X[self.slices[-1]].unstack().T
        self.columns = format.columns
        self.index = format.index  # dataset row major

        # FIXME: .T is depreciated for more than two dimensions
        sliced = torch.tensor(np.array([X[sl].unstack().T.values for sl in self.slices]))
        return torch.swapaxes(sliced, 0, 1)
