import torch
import torch.nn as nn


class FidelityLSTM(nn.Module):
    # TODO @Aditya consider, if the lstm is a plain one, to simply move it into the class below
    #  and remove the readout, as it is in the second mlp
    def __init__(self, input_dim, hidden_dim, layer_dim):
        """
        Basic implementation of a an LSTM network and the readout module to convert the
        hidden dimension to the output dimension

        Args:
            input_dim   : Dimension of the input
            hidden_dim  : Dimension of the hidden state
            layer_dim   : Number of layers
        """
        super(FidelityLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layer of the network
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True,  # We work with tensors, not tuples
        )

        # Readout layer to convert hidden state output to the final output
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
        # FIXME: @Aditya: move this part to the IMFAS1 class
        # Convert the last element of the lstm into values that
        # can be ranked
        out = self.readout(out[:, -1, :])

        return out


class IMFAS_WP(nn.Module):
    def __init__(
            self,
            encoder,
            lstm,
            decoder,
            device: str = "cpu",
    ):
        """
        Workshop paper version of the IMFAS model: https://arxiv.org/pdf/2206.03130.pdf
        MLP1(D) = h_0 -> for_{i=0,..k}  LSTM(h_i, f_i)=h_{i+1}  -> MLP2(h_k) = output

        D: dataset meta-features
        f_i: performances of all n (=algo_dim) algorithms on the i-th fidelity

        The dimensions of the model are the following:
        MLP1: input_dim -> mlp1_hidden_dims -> h_dim
        LSTM: h_dim -> h_dim
        MLP2: h_dim -> mlp2_hidden_dims -> algo_dim

        Args:

            device: device to run the model on

        """

        super(IMFAS_WP, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.fidelity_lstm = lstm

        self.device = torch.device(device)

        self._build_network()

    # def _build_network(self):
    #     """
    #     Build the network based on the initialized hyperparameters
    #
    #     """
    #
    #     # Dataset Meta Feature Encoder:
    #     mlp_dims = [self.meta_features_dim, *self.mlp_hidden_dims, self.algo_dim]
    #     self.encoder = MLP(mlp_dims, activation="relu")
    #
    #     # Fidelity Contextualizer:
    #     self.seq_network = FidelityLSTM(
    #         input_dim=self.lstm_hidden_dim,
    #         hidden_dim=self.mlp_hidden_dims[-1],
    #         layer_dim=self.lstm_layers,
    #         output_dim=self.lstm_hidden_dim,
    #         readout=None,
    #     )
    #
    #     if self.readout:
    #         self.decoder = MLP()

    def forward(self, dataset_meta_features, learning_curves, lc_values_observed):
        """
        Forward path through the meta-feature ranker

        Args:
            dataset_meta_features: tensor of shape (batch_dim, meta_features_dim)
            learning_curves: tensor of shape (batch_dim, n_learning_curves)
            lc_values_observed: if the learning curve values are observed

        Returns:
            tensor of shape (batch_dim, algo_dim)
        """

        initial_hiddenstate = self.encoder(dataset_meta_features)
        # TODO @Aditya can you check that the fidelities are passed in the correct way?
        lstm_D = self.fidelity_lstm(init_hidden=initial_hiddenstate, context=learning_curves)

        return self.decoder(lstm_D)

        return initial, lstm_D


if __name__ == "__main__":
    network = IMFAS_WP()

    # print the network

    print("shared network", network.encoder)
    print("lstm_net", network.seq_network)
