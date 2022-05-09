from typing import List

import torch
import torch.nn as nn
import torchsort


# from ott.tools.soft_sort import ranks as ott_ranks

# TODO shared combiner can be a differential sorting instead of being n MLP

class AlgoRankMLP(nn.Module):
    def __init__(
            self,
            input_dim: int = 107,
            algo_dim: int = 58,
            hidden_dims: List[int] = [300, 200, 100],
            device: torch.device = torch.device("cpu"),
    ):
        super(AlgoRankMLP, self).__init__()
        self.meta_features_dim = input_dim
        self.algo_dim = algo_dim
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

        modules.append(nn.Linear(input_dim, self.algo_dim))
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


class Join_WAVG(nn.Module):
    def __init__(self, n_weights, vec_dim):
        """
        Softmax "normalized and broadcased weights
        :param n_weights number of learnable inputs
        :vec_dim: number of algorithms (dim of X)
        """
        super(Join_WAVG, self).__init__()
        self.shape = vec_dim, n_weights
        self.weights = nn.Parameter(
            torch.distributions.Uniform(0., 1.).sample((n_weights, 1, 1))
        )

        self.softmax = nn.Softmax(dim=1)

    def __repr__(self):
        return f'Join WAVG:\n\tLearned weights: {self.weights.data}'

    def forward(self, X):
        s = self.softmax(self.weights)
        return (s * X).sum(dim=0)


class Join_AVG(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Join_AVG, self).__init__()

    def forward(self, X):
        return X.mean(dim=0)


class AlgoRankMLP_Ensemble(nn.Module):
    joins = {  # fixme: refactor this as configurables that are instantiated and passed to the
        # model?
        'avg': Join_AVG,
        'wavg': Join_WAVG
    }

    def __init__(
            self,
            input_dim: int = 107,
            algo_dim: int = 58,
            shared_hidden_dims: List[int] = [300, 200],
            n_fidelities: int = 3,
            multi_head_dims: List[int] = [100],
            fc_dim: List[int] = [58],
            join: str = 'wavg',
            device: str = 'cpu',
    ):
        """
        Ensemble fo MLPs to rank based on multiple fidelities

        Args:
            input_dim: input dimension
            algo_dim: number of algorithms
            shared_hidden_dims: list of hidden dimensions for the shared MLP
            n_fidelities: number of fidelities
            multi_head_dims: list of hidden dimensions for each multi-head
            fc_dim: list of hidden dimensions for the FC layers
            joint: options: 'avg', plain average of rank outputs
            'wavg' learnable weighted combination of model mlp outputs
            device: device to run the model on

        """

        super(AlgoRankMLP_Ensemble, self).__init__()
        self.meta_features_dim = input_dim
        self.algo_dim = algo_dim
        self.shared_hidden_dims = shared_hidden_dims
        self.multi_head_dims = multi_head_dims
        self.fc_dim = fc_dim
        self.device = torch.device(device)
        self.join = join
        self.n_fidelities = n_fidelities
        self.n_multiheads = n_fidelities - 1

        # self.rank = torchsort.soft_rank

        self._build_network()

    def _build_network(self):
        """
        Build the network based on the initialized hyperparameters

        """

        # Build the shared network
        self.shared_network = AlgoRankMLP(
            input_dim=self.meta_features_dim,
            algo_dim=self.shared_hidden_dims[-1],
            hidden_dims=self.shared_hidden_dims[:-1]
        )

        # TODO Parallelize these networks
        # Build a list of multi-head networks, one for each fidelity
        self.multi_head_networks = nn.ModuleList()

        for _ in range(self.n_multiheads):
            self.multi_head_networks.append(
                AlgoRankMLP(
                    input_dim=self.shared_hidden_dims[-1],
                    algo_dim=self.algo_dim,
                    hidden_dims=self.multi_head_dims,  # fixme: why does this have to be a list
                    # for smac to work
                )
            )

        self.joint = self.joins[self.join](
            n_weights=self.n_multiheads,
            vec_dim=self.algo_dim
        )

        # Build the final network
        self.final_network = AlgoRankMLP(
            input_dim=self.algo_dim,
            algo_dim=self.algo_dim,
            hidden_dims=self.fc_dim,
        )

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
        # TODO Parallelize using joblib
        multi_head_D = []
        for idx in range(self.n_multiheads):
            multi_head_D.append(self.multi_head_networks[idx](shared_D))

        shared_op = self.joint(torch.stack(multi_head_D, dim=0))

        # Forward through the final network
        final_D = self.final_network(shared_op)

        return shared_D, multi_head_D, final_D


if __name__ == "__main__":
    network = AlgoRankMLP_Ensemble()

    # print the network

    print('shared network', network.shared_network)
    print('multi-head', network.multi_head_networks)
    print('final', network.final_network)
