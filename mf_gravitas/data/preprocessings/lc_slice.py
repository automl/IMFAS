import torch

from mf_gravitas.data.preprocessings.table_transforms import Transform


# Since LC datasets are inherently pandas multi-index dataframes,
# a short indexing tutorial can be found here:
# https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

class LC_TimeSlices(Transform):
    def __init__(self, slices: list[int]):
        """
        :param slices: which time slices across algos & dataset you want to have.
        """
        super().__init__(None)
        self.slices = slices

    def fit(self, X):
        """
        :param X: is an Array, where the time dimension is #fixme ???
        :return: torch.Tensor: (n_datasets, n_slices, n_algorithms)
        # fixme: downstream must be tensor!
        """
        # X[self.slices]
        format = X[self.slices[-1]].unstack().T
        self.columns = format.columns
        self.index = format.index  # dataset row major

        sliced = torch.tensor([X[sl].unstack().T.values for sl in self.slices])
        return torch.swapaxes(sliced, 0, 1)
