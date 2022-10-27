from itertools import chain
from typing import List

from torch import nn as nn


class MLP(nn.Module):
    activation = {"relu": nn.ReLU, "identity": nn.Identity}

    def __init__(self, hidden_dims: List[int], activation: str = "relu", readout=False):
        super(MLP, self).__init__()
        self.hidden_dims = hidden_dims

        dims = hidden_dims[:-1] if readout else hidden_dims

        mlp = [
            (
                nn.Linear(in_dim, out_dim, bias=True),
                self.activation[activation](),
                # nn.BatchNorm1d(out_dim
            )
            for in_dim, out_dim in zip(dims, dims[1:])
        ]

        self.layers = nn.ModuleList(chain(*mlp))
        # self.double() # FIXME needed for LSTM?

        if readout:
            self.layers.append(nn.Linear(hidden_dims[-2], hidden_dims[-1], bias=False))

    def forward(self, X):

        print('fwd:')
        for layer in self.layers:
            print(layer, 'input', X)

            X = layer(X)

        return X


if __name__ == "__main__":
    import torch

    in_dim = 4
    mlp = MLP([in_dim, 3, 2, 1], activation="relu")
    batch = 10
    mlp(torch.randn((batch, in_dim)))

    mlp = MLP([in_dim, 3, 2, 1], activation="relu", readout=True)
