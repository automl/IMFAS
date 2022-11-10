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


        # fixme: do we want to have a separate mlp for the lc_encoding, before joining?
        return self.decoder(torch.cat((lc_encoding, dataset_metaf_encoding), 1))


class IMFASTransformerSubsequent(IMFASTransformerMLP):
    def __init__(self, transformer_algos, *args, **kwargs):
        super(IMFASTransformerSubsequent, self).__init__(*args, **kwargs)
        self.transformer_algos = transformer_algos

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
