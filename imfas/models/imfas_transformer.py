from typing import Optional, Tuple

import math
import torch
import torch.nn as nn

import pdb


class PositionalEncoding(nn.Module):
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
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        ## What is happening here
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

        # print('PositionalEncoding -- init')
        # pdb.set_trace()

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
        
        # print('PositionalEncoding')
        # pdb.set_trace() 
        
        if pos_idx is None:
            x = x + self.pe[:, : x.size(1), :]
        else:
            x = x + self.pe[:, pos_idx[0] : pos_idx[1], :]  # type: ignore[misc]
        return self.dropout(x)


class AbstractIMFASTransformer(nn.Module):
    def __init__(
        self,
        n_algos: int,
        n_fidelities: int,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer_layer: torch.nn.TransformerEncoderLayer,
        n_layers=2,
        has_reduce_layer=False,
        device: str = "cpu",
    ):
        """
        Abstract IMFAS Transformer models. Here I propose two types of Transformer models. The first one is to consider
        all the learning curve values of different sequences as features and apply a single forward pass to the tensors.
        The second one is to

        Args:
            encoder:
            decoder:
            transformer_layer:
            n_layers:
            device:
        """
        super(AbstractIMFASTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = transformer_layer.linear1.in_features
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, n_layers)
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.lc_proj_layer = self.build_lc_embedding_layer(n_algos, self.d_model)
        self.positional_encoder = PositionalEncoding(d_model=self.d_model)

        self.to(device)
        self.has_reduce_layer = has_reduce_layer

        if self.has_reduce_layer:
            self.reduce_layer = torch.nn.Linear(n_fidelities + 1, 1)

        # for k, i in self.named_parameters():
        #     print(k, i.device)

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        raise NotImplementedError

    def preprocessing_lcs(self, learning_curves, lc_values_observed):
        (batch_size, n_algos, lc_length) = learning_curves.shape
        
        learning_curves = learning_curves.transpose(1, 2)
        
        lc_values_observed = lc_values_observed.transpose(1, 2)
        
        # print('Abstract Transformer -- preprocessing_lcs')
        # pdb.set_trace()
        
        return learning_curves, lc_values_observed, (batch_size, n_algos, lc_length)

    def embeds_lcs(self, learning_curves, lc_values_observed):
        
        # learning_curves_embedding = self.positional_encoder(learning_curves)
        
        learning_curves_embedding = self.positional_encoder(self.lc_proj_layer(learning_curves))
        

        # print('Abstract Transformer --embeds_lcs')
        # pdb.set_trace()
        
        return learning_curves_embedding, lc_values_observed

    def forward(self, learning_curves, mask, dataset_meta_features=None):
        # FIXME: Consider How to encode Missing values here

        learning_curves, lc_values_observed, lc_shape_info = self.preprocessing_lcs(learning_curves, mask)

        learning_curves_embedding, lc_values_observed = self.embeds_lcs(learning_curves, lc_values_observed)

        encoded_lcs = self.encode_lc_embeddings(learning_curves_embedding, lc_values_observed, lc_shape_info)

        if dataset_meta_features is None:
            dataset_meta_features = torch.full([0], fill_value=0.0, dtype=encoded_lcs.dtype, device=encoded_lcs.device)

            encoded_D = self.encoder(dataset_meta_features).repeat([*encoded_lcs.shape[:-1], 1])
        else:
            encoded_D = self.encoder(dataset_meta_features)
            if len(encoded_lcs.shape) == 3:
                encoded_D = encoded_D.unsqueeze(1).repeat(1, encoded_lcs.shape[1], 1)
        
        # print('AbstractIMFASTransformer -- forward')
        # pdb.set_trace()
        return self.decoder(torch.cat((encoded_lcs, encoded_D), -1)).squeeze(-1)

    def encode_lc_embeddings(
        self,
        learning_curves_embedding: torch.Tensor,
        lc_values_observed: torch.Tensor,
        lc_shape_info: Tuple[int, int, int],
    ) -> torch.Tensor:
        raise NotImplementedError


class IMFASBaseTransformer(AbstractIMFASTransformer):
    """
    This implementation follow the IMFAS's LSTM's implementation, switching the
    LSTM layer with a Transformer layer and being ignorant to HPs, as they are constant.
    """

    def __init__(
        self,
        n_algos: int,
        n_fidelities: int,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer_layer: nn.Module,
        n_layers=2,
        device: str = "cpu",
        has_reduce_layer=False,
    ):
        super(IMFASBaseTransformer, self).__init__(n_algos, n_fidelities, encoder, decoder, transformer_layer, n_layers, has_reduce_layer, device)
        # This is attached at the end of each LCs to indicate that the LC ends here and we could extrac their
        # corresponding feature values. Here I simply compute the number of observed values as an input
        # TODO: Alternative: different embeddings w.r.t. position or algos
        self.lc_length_embedding = nn.Linear(n_algos, transformer_layer.linear1.in_features)

        self.to(torch.device(device))

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        return nn.Linear(n_algos, d_model_transformer)

    def encode_lc_embeddings(self, learning_curves_embedding, mask, lc_shape_info):
        lc_length_embedding = self.lc_length_embedding(mask.sum(1)).unsqueeze(1)

        # lc_values_observed = lc_values_observed.transpose(1, 2)
        
        # print('Base Transformer  -- encode_lc_embeddings')
        # pdb.set_trace()

        # NOTE Transformer encoder should not have the necessity for this
        # encoded_lcs = self.transformer_encoder(learning_curves_embedding)

        encoded_lcs = self.transformer_encoder(
            torch.cat([learning_curves_embedding, lc_length_embedding], dim=1),
        )
        if self.has_reduce_layer:
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2)
            encoded_lcs = self.reduce_layer(encoded_lcs)
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2).squeeze(1)
        else:
            encoded_lcs = encoded_lcs[:, -1, :]
        
        return encoded_lcs


class IMFASHierarchicalTransformer(AbstractIMFASTransformer):
    """
    An IMFAS Trasnforemr with hierarchical architecture. For the input, we first flatten them to perform local attention
    operation with self.transformer_encoder. Then in the second stage, we perform a global attention operation with
    self.global_transformer_encoder
    """

    def __init__(
        self,
        n_algos: int,
        n_fidelities: int,
        encoder: nn.Module,
        decoder: nn.Module,
        transformer_layer: nn.Module,
        n_layers=2,
        device: str = "cpu",
        has_reduce_layer=False,
        pe_on_global_level=False,
    ):
        self.EOS = torch.tensor(0, device=torch.device(device))  # End of Sequence

        super(IMFASHierarchicalTransformer, self).__init__(
            n_algos, n_fidelities, encoder, decoder, transformer_layer, n_layers, has_reduce_layer, device
        )

        self.global_transformer = nn.TransformerEncoder(transformer_layer, n_layers)

        self.eos_embedding_layer = torch.nn.Embedding(2, transformer_layer.linear1.in_features)

        if self.has_reduce_layer:
            self.reduce_layer = torch.nn.Linear(n_algos, 1)

        self.pe_on_global_level = pe_on_global_level

        self.to(torch.device(device))

    def to(self, device):
        self.EOS = self.EOS.to(device)
        super(IMFASHierarchicalTransformer, self).to(device)

    def build_lc_embedding_layer(self, n_algos: int, d_model_transformer: int):
        return nn.Linear(1, d_model_transformer)

    def preprocessing_lcs(self, learning_curves, lc_values_observed):
        lc_shape_info = learning_curves.shape
        batch_size, n_algos, lc_length = learning_curves.shape

        learning_curves = learning_curves.view(batch_size * n_algos, lc_length, 1)
        lc_values_observed = lc_values_observed.view(batch_size * n_algos, lc_length)
        return learning_curves, lc_values_observed, lc_shape_info

    def embeds_lcs(self, learning_curves, lc_values_observed):
        lc_embeddings = self.positional_encoder(self.lc_proj_layer(learning_curves)) * lc_values_observed.unsqueeze(-1)

        n_lcs, lc_length, d_model = lc_embeddings.shape

        # We attach the ending Embedding to the end of each sequences
        eos_embedding = self.eos_embedding_layer(self.EOS)
        lc_embeddings = torch.cat(
            [lc_embeddings, torch.zeros((n_lcs, 1, d_model), dtype=lc_embeddings.dtype, device=lc_embeddings.device)],
            dim=1,
        )

        n_observed_lcs = lc_values_observed.sum(1).long()
        lc_embeddings[torch.arange(n_lcs), n_observed_lcs.long()] = eos_embedding

        # we have an additional item
        lc_values_observed = torch.cat(
            [
                lc_values_observed,
                torch.ones(
                    (len(lc_values_observed), 1), dtype=lc_values_observed.dtype, device=lc_values_observed.device
                ),
            ],
            dim=1,
        )

        return lc_embeddings, lc_values_observed

    def encode_lc_embeddings(self, learning_curves_embedding, lc_values_observed, lc_shape_info):
        batch_size, n_algos, lc_length = lc_shape_info
        n_observed_lcs = lc_values_observed.sum(1).long() - 1

        encoded_lcs_local = self.transformer_encoder(
            learning_curves_embedding, src_key_padding_mask=~lc_values_observed.bool()
        )

        encoded_lcs_local = encoded_lcs_local[torch.arange(len(encoded_lcs_local)), n_observed_lcs]

        encoded_lcs_local = encoded_lcs_local.view(batch_size, n_algos, -1)
        if self.pe_on_global_level:
            encoded_lcs_local = self.positional_encoder(encoded_lcs_local)
        # TODO adjust the Meta features with this type of transformation

        encoded_lcs = self.global_transformer(encoded_lcs_local)
        if self.has_reduce_layer:
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2)
            encoded_lcs = self.reduce_layer(encoded_lcs)
            encoded_lcs = torch.transpose(encoded_lcs, 1, 2).squeeze(1)

        return encoded_lcs
