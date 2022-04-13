from abstract_transform import Transform


class LC_TimeSlice(Transform):
    def __init__(self, slice):
        """
        :param slice: which time slice across algos & dataset you want to have.
        """

    def fit(self, X):
        """
        :param X: is an Array, where the time dimension is  # fixme: which dimensin?
        """
        return self.forward(X)

    def forward(self, X):
        return X[None]
