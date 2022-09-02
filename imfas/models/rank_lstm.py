"""
Main implementation of the Workshop version of IMFAS. AlgoRankMLP and RankLSTM
are the core components of the RankLSTM_Ensemble model (which in fact is the Imfas model)
"""

from typing import List

import torch
import torch.nn as nn
import torchsort


class AlgoRankMLP(nn.Module):
    # FIXME: @Aditya: What does this model do?
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


class RankLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, readout=None):
        """
        Basic implementation of a an LSTM network

        Args:
            input_dim   : Dimension of the input
            hidden_dim  : Dimension of the hidden state (hidden dimensions)
            layer_dim   : Number of hidden layers
            output_dim  : Dimension of the output
            readout     : Optional readout layer for decoding the hidden state
        """
        super(RankLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer of hte network
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,  # We work with tensors, not tuples
        )

        # Readout layer to convert hidden state output to the final output

        if readout is None:
            self.readout = nn.Linear(hidden_dim, output_dim)
        else:
            self.readout = readout

        self.double()

    def forward(self, init_hidden, context):
        """
        Forward pass of the LSTM
        Args:
            init_hidden : Initializer for hidden and/or cell state
            context     : Input tensor of shape (batch_dim, seq_dim, feature_dim)

        Returns:
            Output tensor of shape (batch_dim, output_dim) i.e. the  output readout of the LSTM for each element in a batch
        """

        # Initialize hidden state with the output of the preious MLP
        h0 = torch.stack([init_hidden for _ in range(self.layer_dim)]).requires_grad_().double()

        # Initialize cell state with 0s
        c0 = (
            torch.zeros(
                self.layer_dim,  # Number of layers
                context.shape[0],  # batch_dim
                self.hidden_dim,  # hidden_dim
            )
            .requires_grad_()
            .double()
        )

        # Feed the context as a batched sequence so that at every rollout step, a fidelity
        # is fed as an input to the LSTM
        out, (hn, cn) = self.lstm(context.double(), (h0, c0))
        # FIXME: @Aditya: move this part to the RankLSTM_Ensemble class
        # Convert the last element of the lstm into values that
        # can be ranked
        out = self.readout(out[:, -1, :])

        return out


class RankLSTM_Ensemble(nn.Module):  # FIXME: @Aditya rename: IMFAS?
    def __init__(
            self,
            input_dim: int = 107,
            algo_dim: int = 58,
            # lstm_hidden_dims: List[int] = 100,
            lstm_layers: int = 2,
            shared_hidden_dims: List[int] = [300, 200],
            device: str = "cpu",
    ):
        """
        Sequential Ensemble of LSTM cells to rank based on multiple fidelities

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

        super(RankLSTM_Ensemble, self).__init__()
        self.meta_features_dim = input_dim
        self.algo_dim = algo_dim
        self.lstm_layers = lstm_layers
        self.shared_hidden_dims = shared_hidden_dims
        self.device = torch.device(device)

        self._build_network()

    def _build_network(self):
        """
        Build the network based on the initialized hyperparameters

        """

        # Dataset Meta Feature Encoder:
        self.encoder = AlgoRankMLP(
            input_dim=self.meta_features_dim,
            algo_dim=self.shared_hidden_dims[-1],
            hidden_dims=self.shared_hidden_dims[:-1],
        )

        # Fidelity Contextualizer:
        self.seq_network = RankLSTM(
            input_dim=self.algo_dim,
            hidden_dim=self.shared_hidden_dims[-1],
            layer_dim=self.lstm_layers,
            output_dim=self.algo_dim,
            readout=None,
        )

        # FIXME: @Aditya: there should be a decoder for the final state, before
        #

    def forward(self, dataset_meta_features, fidelities):
        """
        Forward path through the meta-feature ranker

        Args:
            D: input tensor

        Returns:
            algorithm values tensor
        """

        # Forward through the shared network
        shared_D = self.encoder(dataset_meta_features)

        # Forward through the lstm networks to get the readouts
        lstm_D = self.seq_network(init_hidden=shared_D, context=fidelities)

        return shared_D, lstm_D


if __name__ == "__main__":
    network = RankLSTM_Ensemble()

    # print the network

    print("shared network", network.encoder)
    print("lstm_net", network.seq_network)
