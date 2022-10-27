import torch

from imfas.utils.modelinterface import ModelInterface


class RandomBaseline(ModelInterface):
    def __init__(self):
        pass

    def forward(self, learning_curves: torch.Tensor, **kwargs):
        # find the number of algorithms from the shape of the learning curves
        n_algos = learning_curves.shape[1]

        # return random performance for each algorithm
        return torch.rand(n_algos).view(1, -1)
