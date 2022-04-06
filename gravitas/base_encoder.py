import torch
import torch.nn as nn

from typing import List, Tuple, Any

from abc import abstractmethod

class BaseEncoder(nn.Module):
    """
    Base class for encoders thtat are used to get latent representations
    of the data and the algorithms.    
    """
    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()

    def _build_network(self) -> None:
        """
        Bulid the network.
        """
        raise NotImplementedError

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the context.
        """
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Decodes the context.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, D) -> torch.Tensor:
        """
        Forward path through the network to get the encoding
        """
        pass

    @abstractmethod
    def loss_gravity(self) -> torch.Tensor:
        """
        Loss function for gravity based training.
        """
        pass

    @abstractmethod
    def predict_algorithms(self) -> torch.Tensor:
        """
        Predict the algorithms
        """
        pass
        



