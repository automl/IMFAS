from abc import abstractmethod

import torch


class ModelInterface:

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
