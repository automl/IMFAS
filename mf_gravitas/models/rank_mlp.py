from typing import List

import torch
import torch.distributions as td
import torch.nn as nn

import torchsort

import pdb

#from ott.tools.soft_sort import ranks as ott_ranks

#TODO shared combiner can be a differential sorting instead of being n MLP


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
        
        self.rank = torchsort.soft_rank
        
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
        pred = self.rank(pred, **ts_kwargs)
        target = self.rank(target, **ts_kwargs)

        # normalize the soft ranks
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()

        # compute the loss
        spear_loss =  (pred * target).sum()

        return spear_loss.abs()


class ActionRankMLP_Ensemble(nn.Module):
    def __init__(
        self,
        input_dim: int = 107,
        action_dim: int = 58,
        shared_hidden_dims: int = [300, 200], 
        n_fidelities: int = 2,
        multi_head_dims: int = [100], 
        fc_dim: int = [58],
        device: torch.device = torch.device("cpu"),
    ):
        """
        Ensemble fo MLPs to rank based on multiple fidelities

        Args:
            input_dim: input dimension
            action_dim: number of algorithms
            shared_hidden_dims: list of hidden dimensions for the shared MLP
            n_fidelities: number of fidelities
            multi_head_dims: list of hidden dimensions for each multi-head
            fc_dim: list of hidden dimensions for the FC layers
            device: device to run the model on

        """

        super(ActionRankMLP_Ensemble, self).__init__()
        self.meta_features_dim = input_dim
        self.action_dim = action_dim
        self.shared_hidden_dims = shared_hidden_dims
        self.multi_head_dims = multi_head_dims
        self.fc_dim = fc_dim
        self.device = device
        self.n_fidelities = n_fidelities

        self.rank = torchsort.soft_rank

        self._build_network()


    def _build_network(self):
        """
        Build the network based on the initialized hyperparameters

        """

        # Build the shared network
        shared_modules = []

        input_dim = self.meta_features_dim

        for h_dim in self.shared_hidden_dims:
            shared_modules.append(nn.Linear(input_dim, h_dim))
            shared_modules.append(nn.ReLU())
            input_dim = h_dim

        self.shared_network = torch.nn.Sequential(*shared_modules)


        # build a list of multi-head networks, one for each fidelity
        self.multi_head_networks = []

        for _ in range(self.n_fidelities):
            multi_head_modules = [] 
            
            input_dim = self.shared_hidden_dims[-1]
            for h_dim in self.multi_head_dims:
                multi_head_modules.append(nn.Linear(input_dim, h_dim))
                multi_head_modules.append(nn.ReLU())
                input_dim = h_dim

            multi_head_modules.append(nn.Linear(input_dim, self.action_dim))
            multi_head_modules.append(nn.ReLU())

            self.multi_head_networks.append(torch.nn.Sequential(*multi_head_modules))

        self.multi_head_network = nn.Sequential(*self.multi_head_networks)

        # Build the final network
        final_modules = []
        input_dim = self.action_dim
        for h_dim in self.fc_dim:
            final_modules.append(nn.Linear(input_dim, h_dim))
            final_modules.append(nn.ReLU())
            input_dim = h_dim

        final_modules.append(nn.Linear(input_dim, self.action_dim))
        final_modules.append(nn.ReLU())

        self.final_network = torch.nn.Sequential(*final_modules)




    def forward(self, D):
        """
        Forward path through the meta-feature ranker

        Args:
            D: input tensor

        Returns:
            algorithm values tensor
        """


        # Forward through the shared network
        shared_D = self.shared_network(D)

        # Forward through the multi-head networks
        multi_head_D = []
        for idx in range(self.n_fidelities):
            multi_head_D.append(self.multi_head_networks[idx](shared_D))
        
        # Average the outputs

        shared_op = shared_D.mean(dim=0)
        
        # Forward through the final network
        final_D = self.final_network(shared_op)

        return shared_D, multi_head_D, final_D



    def pred_loss(self, pred, target,  **ts_kwargs):
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
        pred = self.rank(pred, **ts_kwargs)
        target = self.rank(target, **ts_kwargs)

        # normalize the soft ranks
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        target = target - target.mean()
        target = target / target.norm()

        # compute the loss
        spear_loss =  (pred * target).sum()

        return spear_loss.abs() 




if __name__ == "__main__":
    

    network = ActionRankMLP_Ensemble()

    #print(network)

    # print the network

    print(network.shared_network)
    print(network.multi_head_networks)
    print(network.final_network)


