from typing import List

import torch
import torch.nn as nn
import torchsort
from torch.nn import Linear, LSTM

from mf_gravitas.models.rank_mlp import AlgoRankMLP

import pdb

class RankLSTM(nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            layer_dim, 
            output_dim, 
            readout=None
        ):
        '''
        Basic implementation of a an LSTM network
        Args:
            input_dim: Dimension of the input
            hidden_dim: Dimension of the hidden state
            layer_dim: Number of layers
            output_dim: Dimension of the output
            readout: Optional readout layer for decoding the hidden state
        '''
        super(RankLSTM, self).__init__()
        
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # LSTM layer of hte network
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
                        input_size = input_dim, 
                        hidden_size= hidden_dim, 
                        num_layers= layer_dim, 
                        batch_first=True   # We work with tensors, not tuples
                    )

        # Readout layer to convert hidden state output to the final output
        
        if readout is None:
            self.readout = nn.Linear(
                            hidden_dim, 
                            output_dim
                        )
        else:
            self.readout = readout

    def forward(self, x):
        '''
        Forward pass of the LSTM
        Args:
            x: Input tensor of shape (batch_dim, seq_dim, feature_dim)
        Returns:
            Output tensor of shape (batch_dim, output_dim) i.e. the  output readout of the LSTM for each element in a batch 
        '''
        
        # TODO test other initializations

        x = torch.reshape(x, (1, x.shape[0], x.shape[1]))
        # print(x.shape)
        # pdb.set_trace()


        

        # Initialize hidden state with zeros
        h0 = torch.zeros(
                self.layer_dim,     # Number of layers
                x.shape[0],          # batch_dim
                self.hidden_dim     # hidden_dim
            ).requires_grad_()

        # Initialize cell state with 0s
        c0 = torch.zeros(
                self.layer_dim,     # Number of layers
                x.shape[0],          # batch_dim  
                self.hidden_dim     # hidden_dim
            ).requires_grad_()

        # detach the gates because we do truncated BPTT
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step and pass that to the output readout
        out = self.readout(out[:, -1, :]) 

        return out



class RankLSTM_Ensemble(nn.Module):

    def __init__(
            self,
            input_dim: int = 107,
            algo_dim: int = 58,
            lstm_hidden_dims: List[int] = [100, 100],
            lstm_layers: int = [2, 2],
            shared_hidden_dims: List[int] = [300, 200],
            fc_dim: List[int] = [58],
            n_fidelities: int = 3,   
            sequential: bool = True,         
            device: str = 'cpu',
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
        self.lstm_hidden_dims = lstm_hidden_dims
        self.lstm_layers = lstm_layers
        self.shared_hidden_dims = shared_hidden_dims
        self.fc_dim = fc_dim
        self.device = torch.device(device)

        self.n_fidelities = n_fidelities

        # Basic assertions for lstms
        assert(len(self.lstm_hidden_dims) == len(self.lstm_layers))
        assert(len(self.lstm_hidden_dims) == n_fidelities - 1)

        self.sequential  = sequential

        self.rank = torchsort.soft_rank

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


        # Build the lstms
        self.lstm_network = nn.ModuleList()
        ipd = self.shared_hidden_dims[-1]
        for i in range(len(self.lstm_hidden_dims)):
            self.lstm_network.append(
                RankLSTM(
                    input_dim=ipd,
                    hidden_dim=self.lstm_hidden_dims[i],
                    layer_dim=self.lstm_layers[i],
                    output_dim=self.algo_dim,
                    readout=None
                )
            )

            if self.sequential:
                ipd = self.algo_dim

        # self.lstm_network = torch.nn.Sequential(*self.lstm_network)

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

        # Forward through the lstm networks to get the readouts
        lstm_D = []
        ip = shared_D

        for i in range(len(self.lstm_network)):
        
            op = self.lstm_network[i](ip)
            lstm_D.append(op)
            
            if self.sequential:
                ip = op

    
        # Forward through the final network    
        if self.sequential:
            final_D = self.final_network(lstm_D[-1])
        else:
            final_D = self.final_network(torch.stack(lstm_D).mean(dim=0))

        return shared_D, lstm_D, final_D


if __name__ == "__main__":
    network = RankLSTM_Ensemble()

    # print the network

    print('shared network', network.shared_network)
    print('lstm_net', network.lstm_network)
    print('final', network.final_network)
