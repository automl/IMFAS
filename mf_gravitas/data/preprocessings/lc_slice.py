from mf_gravitas.data.preprocessings.table_transforms import Transform


# Since LC datasets are inherently pandas multi-index dataframes,
# a short indexing tutorial can be found here:
# https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe

class LC_TimeSlice(Transform):
    def __init__(self, slice: int):
        """
        :param slice: which time slice across algos & dataset you want to have.
        """
        super().__init__(None)
        self.slice = slice

    def fit(self, X):
        """
        :param X: is an Array, where the time dimension is  # fixme: which dimensin?
        """
        X = X[self.slice].unstack().T
        self.columns = X.columns
        self.index = X.index
        return X
