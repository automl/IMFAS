from typing import Optional

import torch
from torch import nn


# marius proposal: prepend the dataset meta features to the sequence.

class IMFASTransformerMLP(nn.Module):
    def __init__(
            self,

            positional_encoder,
            transformer_lc,
            decoder,
            device,
            dataset_metaf_encoder: Optional[nn.Module] = None,
            **kwargs  # placeholder to make configuration more convenient
    ):
        """This model class defines the computational macro graph of the IMFAS model.
        This variant passes the entire learning curve tensor to the transformer encoder jointly! """
        super(IMFASTransformerMLP, self).__init__()

        self.dataset_metaf_encoder = dataset_metaf_encoder
        self.positional_encoder = positional_encoder

        self.transformer_lc = transformer_lc

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

    def forward_lc(self, learning_curves: torch.Tensor, mask: torch.Tensor, **kwargs):

        # prepend a zero timestep to the learning curve tensor if n_fidelities is uneven
        # reasoning: transformer n_heads must be a devisor of d_model
        if learning_curves.shape[-2] % 2 == 1:
            # fixme: untested code
            learning_curves = torch.cat(
                (torch.zeros_like(learning_curves[:, :, :1]), learning_curves), dim=-2)

            mask = torch.cat(
                (torch.zeros_like(mask[:, :, :1]), mask), dim=-2)

        pos_learning_curves = self.positional_encoder(learning_curves)

        lc_encoding = self.transformer_lc(
            # reshape the learning curve tensor to trick the transformer in believing
            # the algo dim is the batch dim
            # alter mask encoding (since pytorch expects mask that indicates missing values)
            pos_learning_curves.permute(1, 2, 0),

            # same as above regarding permutation. But we need to invert & convert the mask,
            # since pytorch expects mask that indicates missing values & type bool,
            # such that mask.masked_fill(~mask.bool(), float('-inf')) is performed directly before
            # softmax in the attention head
            src_key_padding_mask=~mask.bool().permute(1, 2, 0).view(self.n_algos, self.n_fidelities)
            # make it a
            # 2d mask
        )

        return lc_encoding.permute(0, 1, 2)

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor,
                dataset_meta_features: Optional[torch.Tensor] = None, **kwargs):

        # n_datasets is batch dimension
        self.n_datasets, self.n_algos, self.n_fidelities = learning_curves.shape

        # (dataset_meta_feature encoding is optional) --------------------------
        # TODO @difan: is there a better was to do this (when we want to cmd all datasets with a
        #  single config)?
        if dataset_meta_features is None:
            # if dataset meta features are not available, we use a zero vector.
            self.dataset_metaf_encoding = torch.zeros(
                self.dataset_metaf_encoder.layers[-1].weight.shape[-1]
            ).view(1, -1).to(self.device)

        else:
            # default case: dataset meta features are available
            self.dataset_metaf_encoding = self.dataset_metaf_encoder(dataset_meta_features)

        lc_encoding = self.forward_lc(learning_curves, mask)
        print(lc_encoding)

        # consider: do we want to have a separate mlp for the lc_encoding, before joining?
        return self.decoder(
            # stack flattened lc_encoding with dataset_metaf_encoding
            torch.cat([lc_encoding.view(1, -1), self.dataset_metaf_encoding], 1)
        )


class CustomTransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    """
    This custom transformer encoder layer allows to alter the query vector of the self-attention block.
    The query vector is altered by a linear layer and a ReLU activation, which in turn is
    multiplied with the dataset meta features. (basically piping the encoded dataset meta
    features into the self-attention block to alter the query vector). Rational: the
    dataset_meta_features contain information regarding the scaling of the dataset i.e. context
    to the observed learning curve. So we should alter the attention. We do this by altering the
    query vector.
    """

    def __init__(self, reference_object, n_fidelities, *args, **kwargs):
        """
        :param reference_object: the object that contains the encoded dataset meta features i.e.
        e.g. reference_object.dataset_metaf = mlp(dataset_meta_features)
        where reference_object.dataset_metaf.shape[-1] is the encoding (/mlp output) dimension of
        the dataset meta features
        """
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

        self.reference_object = reference_object
        self.q_layer = nn.Linear(self.reference_object.dataset_metaf.shape[-1], n_fidelities)
        self.q_activation = nn.ReLU()

    def forward(self, *args, **kwargs):
        return super(CustomTransformerEncoderLayer, self).forward(*args, **kwargs)

    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = x * self.q_activation(self.q_layer(self.reference_object.dataset_metaf))
        k = x
        x = self.self_attn(q, k, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
