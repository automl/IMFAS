from typing import Tuple, Optional

import math
import torch
from torch import nn


class PositionalEncoder(nn.Module):
    r"""This function is direcly copied from Auto-PyTorch Forecasting and is a modified version of
        https://github.com/pytorch/examples/blob/master/word_language_model/model.py

        NOTE: different from the raw implementation, this model is designed for the batch_first inputs!
        Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model (int):
            the embed dim (required).
        dropout(float):
            the dropout value (default=0.1).
        max_len(int):
            the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:  # even vs uneven number of dimensions
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos_idx: Optional[Tuple[int]] = None) -> torch.Tensor:
        r"""Inputs of forward function
        Args:
            x (torch.Tensor(B, L, N)):
                the sequence fed to the positional encoder model (required).
            pos_idx (Tuple[int]):
                position idx indicating the start (first) and end (last) time index of x in a sequence

        Examples:
            >>> output = pos_encoder(x)
        """
        if pos_idx is None:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, pos_idx[0]: pos_idx[1], :]  # type: ignore[misc]
        return self.dropout(x)


class IMFAS_Joint_Transformer(nn.Module):
    def __init__(
            self,
            dataset_metaf_encoder,
            positional_encoder,
            transformer_encoder,
            decoder,
            device,
            **kwargs  # placeholder to make configuration more convenient
    ):
        """This model class defines the computational macro graph of the IMFAS model.
        This variant passes the entire learning curve tensor to the transformer encoder jointly! """
        super(IMFAS_Joint_Transformer, self).__init__()

        self.dataset_metaf_encoder = dataset_metaf_encoder
        self.positional_encoder = positional_encoder
        self.transformer_encoder = transformer_encoder
        self.decoder = decoder
        self.device = device

        self.check_conformity()

    def check_conformity(self):
        """This method checks if the model components are configured correctly"""
        n_algos, n_datasets = None, None  # self.transformer_encoder.layers[0]

        output_dim_transformer = None  # self.transformer_encoder.layers[-1]
        output_dim_dmetaf = None  # self.dataset_metaf_encoder.layers[-1]
        decoder_indim = None  # self.decoder.layers[0]

        # assert output_dim_transformer + output_dim_dmetaf == decoder_indim, \
        #    "The output dimension of the transformer encoder and the dataset meta feature encoder must be equal to the input dimension of the decoder!"

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
        dataset_metaf_encoding = self.dataset_metaf_encoder(dataset_meta_features)

        # prepend a zero timestep to the learning curve tensor if n_fidelities is uneven
        # reasoning: transformer n_heads must be a devisor of d_model

        if learning_curves.shape[-1] % 2 == 1:
            learning_curves = torch.cat(
                (torch.zeros_like(learning_curves[:, :, :1]), learning_curves), dim=2)

            mask = torch.cat(
                (torch.zeros_like(mask[:, :, :1]), mask), dim=2)

        pos_learning_curves = self.positional_encoder(learning_curves)

        # TODO check if it shouldn't be src_key_padding_mask instead?
        # fixme: check why the mask dim is not correct

        # CAREFULL: this is what should be happening in attentionhead, to
        # avoid the transformer from peaking into padded values
        # print(mask.masked_fill(~mask.bool(), float('-inf')))
        # but we actually need our mask to be bool and inverted for that!

        lc_encoding = self.transformer_encoder(
            pos_learning_curves,
            mask=mask)

        return self.decoder(torch.cat((lc_encoding, dataset_metaf_encoding), 1))


class IMFAS_LateJoint_Transformer(IMFAS_Joint_Transformer):
    def check_connformaty(self):
        """This method checks if the model components are configured correctly"""
        n_algos, n_datasets = None, None  # self.transformer_encoder.layers[0]

        output_dim_transformer = None  # self.transformer_encoder.layers[-1]
        output_dim_dmetaf = None  # self.dataset_metaf_encoder.layers[-1]
        decoder_indim = None  # self.decoder.layers[0]

        # assert output_dim_transformer + output_dim_dmetaf == decoder_indim, \
        #    "The output dimension of the transformer encoder and the dataset meta feature encoder must be equal to the input dimension of the decoder!"

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
        """To reduce the model complexity, we could also move the learning curves individually
        through the transformer encoder and then concatenate the results before passing them
        through the decoder, to produce a joint interpretation."""

        dataset_meta_features = self.dataset_metaf_encoder(dataset_meta_features)
        pos_encoded_lcs = self.positional_encoder(learning_curves)

        # reshape the learning curves such that they appear to be batched.
        # The transformer will assume them to be independent and do a forward pass for each one.
        # the result needs to be unstacked.
        # fixme: double check copilot's suggestion!
        lc_encoding = self.transformer_encoder(
            pos_encoded_lcs.view(-1, pos_encoded_lcs.shape[2], pos_encoded_lcs.shape[3]),
            mask=mask.view(-1, mask.shape[2])
        ).view(pos_encoded_lcs.shape[0], pos_encoded_lcs.shape[1], -1)

        return self.decoder(torch.cat((lc_encoding, dataset_meta_features), 2))


if __name__ == '__main__':
    from imfas.utils.mlp import MLP
    from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer

    from imfas.data.lcbench.example_data import train_dataset

    n_datasets, n_algos, n_fidelities = train_dataset.learning_curves.shape
    n_dataset_metaf = train_dataset.meta_dataset.shape[1]

    transf_in_dim = n_algos * n_fidelities
    transf_dim_feedforward = 200

    model = IMFAS_Joint_Transformer(
        dataset_metaf_encoder=MLP(),
        positional_encoder=PositionalEncoder(),
        transformer_encoder=TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=transf_in_dim,
                dim_feedforward=transf_dim_feedforward,
                nhead=8
            ),
            num_layers=3,  # n_times, the encoder layer is repeated
            norm=nn.LayerNorm(transf_dim_feedforward),
            enable_nested_tensor=False,  # relevant for high levels of padding!
        ),
        # encoder could have arbitrary intermediate number of layers
        decoder=MLP(hidden_dims=[n_dataset_metaf + transf_dim_feedforward, 100, n_algos]),
    )
    #
    # model = IMFAS_LateJoint_Transformer(
    #     dataset_metaf_encoder=MLP(),
    #     positional_encoder=PositionalEncoder(),
    #     transformer_encoder=TransformerEncoder( # carefull with the parametrization
    #         encoder_layer=TransformerEncoderLayer(),
    #         num_layers=3,
    #         norm=nn.LayerNorm(),
    #         enable_nested_tensor=False,  # relevant for high levels of padding!
    #     ),
    #     decoder=MLP(),
    # )

    idx = 0
    model(**train_dataset[idx][0])
