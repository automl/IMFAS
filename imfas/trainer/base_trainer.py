from typing import Callable, Dict, Optional

import numpy as np
import torch.optim
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm


class BaseTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer = None,
            # callbacks_end: List[Callable] = None
    ):
        """

        :param model: torch.nn.Module
        :param optimizer: A partially instantiated object accepting model.parameters()
        :param callbacks_end: list of functions to be called at the end of each epoch
        """
        self._step = 0  # FIXME: @Aditya: why exactly doesn't it suffice to have the epoch?
        self.model = model
        self.device = self.model.device

        if optimizer is not None:
            self.optimizer = optimizer(self.model.parameters())
        # self.callbacks_end = callbacks_end

    @property
    def step(self):
        return self._step

    def to_device(self, input, ) -> None:
        for k, v in input.items():
            input[k] = v.to(self.device).float()
        return input

    def train(self, train_loader, epochs, loss_fn, log_freq=5) -> None:
        """define one epoch of training"""
        # TODO: incorporate callbacks to the training loop (before and after each epoch, and )
        #  at every (k-th) step?
        for _, data in enumerate(train_loader):
            # Data parsing
            X, y = data  # assuming a dict of tensors here for each
            self.to_device(X)
            self.to_device(y)  # fixme: move to device in fwd call (to allow for data prep such as
            # masking?)

            self.optimizer.zero_grad()

            y_hat = self.model.forward(**X)

            loss = loss_fn(y_hat, y["final_fidelity"])
            # print(y, y_hat, loss)
            loss.backward()

            # FIXME: y needs to be explicit or have a strong conventioN

            # Log the training loss
            if self.step % log_freq == 0:
                wandb.log({"trainingloss": loss}, step=self.step)  # fixme: every training epoch!

            self.optimizer.step()
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

                y_hat = self.model.forward(**X)
                loss = valid_loss_fn(y_hat, y["final_fidelity"])

                losses.append(loss)

        if aggregate_fn is not None:
            return aggregate_fn(losses)  # fixme: we might want an aggregate for each loss fn
        else:
            return losses  # returns the entire trace of all instances in the testloader

    def run(
            self,
            train_loader,
            test_loader,
            epochs,
            train_loss_fn,
            log_freq=5,
            valid_loss_fns: Dict[str, Callable] = None,
            aggregate_fn: Optional[Callable] = None,
    ):
        """Main loop including training & test evaluation, all of which report to wandb"""

        # Class & functional interface
        if isinstance(train_loss_fn, DictConfig):
            train_loss_fn = instantiate(train_loss_fn)

        elif isinstance(train_loss_fn, Callable):
            pass

        for epoch in tqdm(range(epochs), desc='Training epochs'):
            self.train(train_loader, epoch, train_loss_fn)

            # move this  parameter hist tracker to a callback?
            for k, t in self.model.state_dict().items():
                wandb.log({k: wandb.Histogram(torch.flatten(t))}, step=self.step)

            if valid_loss_fns is not None and self.step % log_freq == 0:
                # End of epoch validation loss tracked in wandb

                for k, fn in valid_loss_fns.items():

                    if isinstance(fn, DictConfig):
                        fn = instantiate(fn)

                    loss = self.evaluate(test_loader, fn, aggregate_fn)
                    # self.losses[k].append(loss)

                    if self.step % log_freq == 0:
                        # Log all the  losses in wandb
                        wandb.log({k: np.mean(loss)}, step=self.step)

            # # execute other callbacks on the end of each epoch
            # for callback in self.callbacks_end:
            #     callback(trainer=self)
