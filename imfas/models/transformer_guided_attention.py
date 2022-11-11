from typing import Optional

import torch
from torch import nn

from imfas.models.transformerMLP import IMFASTransformerMLP
from imfas.utils import MLP


class IMFASTransformerGuidedAttention(IMFASTransformerMLP):

    def __init__(
            self,

            positional_encoder,

            lc_encoder_layer,
            # decoder,
            device,
            decoder_hidden_dims: Optional[list],
            n_algos: int,
            n_fidelities: int,
            dataset_metaf_encoder: Optional[nn.Module] = None,
            lc_encoder_kwargs: Optional[dict] = None,
            **kwargs  # placeholder to make configuration more convenient
    ):
        """This model class defines the computational macro graph of the IMFAS model.
        This variant passes the entire learning curve tensor to the transformer encoder jointly! """
        nn.Module.__init__(self)

        self.dataset_metaf_encoder = dataset_metaf_encoder
        dmetaf_encoder_outdim = self.dataset_metaf_encoder.layers[-2].weight.shape[-2]
        self.positional_encoder = positional_encoder

        encoder_layer = lc_encoder_layer()
        self.transformer_lc = torch.nn.TransformerEncoder(encoder_layer, **lc_encoder_kwargs)

        # make the custom transformer layer aware of self.dataset_metaf_encoding (which the
        # forward places in self every time it is called). This is a hacky way to pass the
        # dataset meta feature encoding into  self attention
        print(id(self))
        for l in self.transformer_lc.layers:
            l.reference_object = self  # maybe instead make it classattribute to
            # CustomTransformerEncoderLayer to avoid recursion?
            print(id(l.reference_object))

        # n_fidelities + 1 for nan safeguard
        decoder_input_dim = dmetaf_encoder_outdim + (n_fidelities + 1) * n_algos
        self.decoder = MLP(hidden_dims=[decoder_input_dim, *decoder_hidden_dims, n_algos])

        self.device = device

    # the CustomTransformerEncoderLayer needs access to self.dataset_metaf
    # which is overwritten at every forward pass to make it accessible here!


class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    This custom transformer encoder layer allows to alter the query vector of the self-attention block.
    The query vector is altered by a linear layer and a ReLU activation, which in turn is
    multiplied with the dataset meta features. (basically piping the encoded dataset meta
    features into the self-attention block to alter the query vector). Rational: the
    dataset_meta_features contain information regarding the scaling of the dataset i.e. context
    to the observed learning curve. So we should alter the attention. We do this by altering the
    query vector.
    """

    def __init__(self, dim_dataset_metaf_encoding, n_fidelities, *args, **kwargs):
        """
        :args reference_object: the object that contains the encoded dataset meta features i.e.
        e.g. reference_object.dataset_metaf = mlp(dataset_meta_features)
        where reference_object.dataset_metaf.shape[-1] is the encoding (/mlp output) dimension of
        the dataset meta features
        """
        super(CustomTransformerEncoderLayer, self).__init__(*args, **kwargs)

        self.reference_object = None
        self.mlp = nn.Sequential(
            nn.Linear(dim_dataset_metaf_encoding, n_fidelities),
            nn.ReLU()
            # todo : consider making subsequent layers configurable
        )

    @property
    def dataset_metaf_encoding(self):
        return self.refrence_object.dataset_metaf_encoding

    def forward(self, *args, **kwargs):
        return nn.TransformerEncoderLayer.forward(self, *args, **kwargs)

    def _sa_block(self, x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        print(id(self.reference_object))
        q = x * self.mlp(self.dataset_metaf_encoding)
        k = x
        x = self.self_attn(q, k, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)


object
