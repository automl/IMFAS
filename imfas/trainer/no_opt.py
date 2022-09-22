'''
    Interim trainer for  algorithm selectors

    NOTE: This can be potentially combined with slice_evaluator
    

'''

from typing import Callable, Dict, Optional

import torch.optim
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from imfas.trainer.base_trainer import BaseTrainer

import pdb

class NoOptTrainer(BaseTrainer):
    def __init__(self, model: torch.nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        
        """

        :param model: torch.nn.Module
        :param optimizer: A partially instantiated object accepting model.parameters()
        :param callbacks_end: list of functions to be called at the end of each epoch
        """
        self._step = 0  # FIXME: @Aditya: why exactly doesn't it suffice to have the epoch?
        self.model = model
        self.device = self.model.device

    @property
    def step(self):
        return self._step

    def to_device(self, input, ) -> None:
        for k, v in input.items():
            input[k] = v.to(self.device).float()
        return input

    def train(self, train_loader) -> None:
        """define one epoch of training"""
        # TODO: incorporate callbacks to the training loop (before and after each epoch, and )
        #  at every (k-th) step?
        for _, data in enumerate(train_loader):
            # Data parsing
            X, y = data  # assuming a dict of tensors here for each

            self.to_device(X)
            self.to_device(y) 

            # forward propagate the create of RFs 
            self.model.fit(**X)                     # TODO implicitly check for training and/or validation in forward

        self._step += 1

    def evaluate(self, test_loader, valid_loss_fn, aggregate_fn=None):
        """evaluate the model on the test set after epoch ends for a single validation function"""
        losses = []
        self.model.eval()  # Consider: since we can ask the state of the model
        # we can actuall use this to change the forward to track particular values

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                X, y = data
                self.to_device(X)
                self.to_device(y)
                

                y_hat = self.model.predict(**X)
                loss = valid_loss_fn(y_hat, y["final_fidelity"][0])

                print(valid_loss_fn, loss)

                losses.append(loss)

        return losses[0] if aggregate_fn is None else aggregate_fn(losses)

    def run(
        self,
        train_loader,
        test_loader,
        valid_loss_fns: Dict[str, Callable] = None,
        aggregate_fn: Optional[Callable] = None,
    ):
        """Main loop including training & test evaluation, all of which report to wandb"""

        #for epoch in tqdm(range(epochs), desc='Training epochs'):

        # Train the model
        self.train(train_loader)

        # Track validation metrics
        if valid_loss_fns is not None:
            # End of epoch validation loss tracked in wandb

            for k, fn in valid_loss_fns.items():

                if isinstance(fn, DictConfig):
                    fn = instantiate(fn)

                loss = self.evaluate(test_loader, fn, aggregate_fn)

                wandb.log({k: loss.item()}, step=self.step)  
