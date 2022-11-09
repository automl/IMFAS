import torch
from torch import nn


class IMFASTransformerMLP(nn.Module):
    def __init__(
            self,
            dataset_metaf_encoder,
            positional_encoder,
            transformer_lc,
            decoder,
            device,
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

        # CAREFULL: this is what should be happening in attentionhead, to
        # avoid the transformer from peaking into padded values
        # print(mask.masked_fill(~mask.bool(), float('-inf')))
        # but we actually need our mask to be bool and inverted for that!

        # reshape the learning curve tensor to trick the transformer in believing
        # the algo dim is the batch dim
        # alter mask encoding (since pytorch expects mask that indicates missing values)
        pos_learning_curves = pos_learning_curves.transpose(0, 1).transpose(1, 2)
        mask = ~mask.bool().transpose(0, 1).transpose(1, 2)

        lc_encoding = self.transformer_encoder(
            pos_learning_curves,
            mask=mask
        )

        return lc_encoding

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
        # n_datasets is batch dimension
        n_datasets, n_algos, n_fidelities = learning_curves.shape

        dataset_metaf_encoding = self.dataset_metaf_encoder(dataset_meta_features)

        lc_encoding = self.forward_lc(learning_curves, mask)

        # stack the lc_encoding into a single vector
        lc_encoding = lc_encoding.reshape(1, -1)  # a single batch dim!

        # fixme: do we want to have a separate mlp for the lc_encoding, before joining?
        return self.decoder(torch.cat((lc_encoding, dataset_metaf_encoding), 1))


class IMFASTransformerSubsequent(IMFASTransformerMLP):
    def __init__(self, transformer_algos, *args, **kwargs):
        super(IMFASTransformerSubsequent, self).__init__(*args, **kwargs)
        self.transformer_algos = transformer_algos

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, dataset_meta_features,
                **kwargs):
