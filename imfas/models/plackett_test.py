import torch.nn as nn


class PlackettTest(nn.Module):
    def __init__(self, encoder: nn.Module, device: str = "cpu"):
        super(PlackettTest, self).__init__()
        self.encoder = encoder
        self.device = device

    def forward(self, dataset_meta_features, learning_curves, *args, **kwargs):
        return self.encoder(dataset_meta_features)
