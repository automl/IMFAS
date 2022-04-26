from typing import List

import torch
import torch.distributions as td
import torch.nn as nn

import torchsort

import pdb

class ActionRankMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 107,
        action_dim: int = 58,
        hidden_dims: List[int] = [300, 200, 100],
        device: torch.device = torch.device("cpu"),
    ):
        
        super(ActionRankMLP, self).__init__()
        self.meta_features_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.device = device
        
        self._build_network()

    def _build_network(self):
        """
        Build the network based on the initialized hyperparameters
        """
        modules = []

        hidden_dims = self.hidden_dims
        input_dim = self.meta_features_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.action_dim))
        modules.append(nn.ReLU())
        
        self.network = torch.nn.Sequential(*modules)
    
    def forward(self, D):
        """
        Forward path through the meta-feature ranker
        
        Args:
            D: input tensor

        Returns:
            algorithm values tensor
        
        """
        return self.network(D)

    
    def loss(self, pred, target,  **ts_kwargs):
        """
        Loss function for the meta-feature ranker


        Args:
            pred: predicted values
            target: target values
            ts_kwargs: keyword arguments for the loss function

        Returns:
            differentiable loss tensor
        """

        # generate soft ranks
        pred = torchsort.soft_rank(pred, **ts_kwargs)
        target = torchsort.soft_rank(target, **ts_kwargs)

        # normalize the soft ranks
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()

        # compute the loss
        spear_loss =  (pred * target).sum()

        return spear_loss.abs()