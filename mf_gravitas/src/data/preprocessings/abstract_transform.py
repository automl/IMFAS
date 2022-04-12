"""
Concept of how a configurable Transformation Pipeline might be look like
(inspired by both sklearn's and torchvision's transofrmations
"""

import torch.nn as nn


class Transform:  # TODO: nn.Module?
    """
    For table based data.

    Similar to sklearn's transforms (e.g. StandardScalar)
    Each transform should have a fit and a transform method.
    # TODO how to chain multiple transformations after one another & reuuse
    #   the fitted & transformed separately?
    """

    def __init__(self, columns: tuple[str]):
        self.columns = columns

    def fit(self, X):
        # get all the necessary statistics, that will be used in both the train and
        # test set - to ensure that the same scales are used for the transforms.
        X[self.columns]

        self.sd = X.sd()

        return self.forward(X)

    def forward(self, X):
        # once the transformer is fit,
        # this method can be called to actually transform
        return X / self.sd


class Pipeline(nn.Sequential):
    def fit(self, input):
        """
        first time around fit & transform on training data. In the
        second run, simply use the nn.Sequentials forward method (is also its call
        to transform a test set to the very same scale!
        """
        for module in self:
            input = module.fit(input)
        return input


if __name__ == '__main__':
    X_train = None
    X_test = None
    pipe = Pipeline(columns=('A', 'B'))
    X_trained_transformed = pipe.fit(X_train)
    X_test_transformed = pipe(X_test)
