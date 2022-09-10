from itertools import chain
from typing import List

from torch import nn as nn


class MLP(nn.Module):
    activation = {
        'relu': nn.ReLU,
        'identiy': nn.Identity
    }

    def __init__(self, mlp_dims: List[int], activation: str = "relu"):
        super(MLP, self).__init__()
        self.mlp_dims = mlp_dims

        mlp = [(nn.Linear(in_dim, out_dim, bias=True), self.activation[activation](),
                nn.BatchNorm1d(out_dim))
               for in_dim, out_dim in zip(mlp_dims, mlp_dims[1:])]

        self.layers = nn.ModuleList(chain(*mlp))
        # self.double() # FIXME needed for LSTM?

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)

        return X


if __name__ == '__main__':
    import torch

    in_dim = 4
    mlp = MLP([in_dim, 3, 2, 1], activation='relu')
    batch = 10
    mlp(torch.randn((batch, in_dim)))
