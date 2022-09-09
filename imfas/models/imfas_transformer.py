from typing import List

import torch
import torch.nn as nn


class IMFASTransformer(torch.nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, transformer_dims: [List],
                 device: str = 'cpu'):
        super(IMFASTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.build_transformerencoder()
        self.device = device

    def build_transformerencoder(self):
        self.transformer_encoder = None

    def forward(self, dataset_meta_features, learning_curves):
        encoded_D = self.encoder(dataset_meta_features)

        encoded_lcs = self.transformer_encoder(learning_curves)

        return self.decoder(torch.cat(((encoded_lcs, encoded_D), 1)))
