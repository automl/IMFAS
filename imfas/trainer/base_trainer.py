from typing import Callable, List, Dict

import torch.optim
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig


class BaseTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            callbacks_end: List[Callable] = None
    ):
        """

        :param model: torch.nn.Module
        :param optimizer: A partially instantiated object accepting model.parameters()
        :param callbacks_end: list of functions to be called at the end of each epoch
        """
        self._step = 0  # FIXME: @Aditya: why exactly doesn't it suffice to have the epoch?
        self.model = model
        self.device = self.model.device

        self.optimizer = optimizer(self.model.parameters())
        self.callbacks_end = callbacks_end

    @property
    def step(self):
        return self._step

    def to_device(self, dict, ) -> None:
        for v in dict.values():
            v.to(self.device)

    def train(self, train_loader, epochs, loss_fn) -> None:
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
            print(X.keys())
            y_hat = self.model.forward(**X)
            loss = loss_fn(y_hat, y['y']).backward()  # FIXME: y needs to be explicit & have a
            # convention in dictionary naming

            wandb.log({'trainingloss': loss}, step=self.step)  # fixme: every training step?

            self.optimizer.step()
            self._step += 1

    def evaluate(self, test_loader, valid_loss_fn, aggregate_fn):
        """evaluate the model on the test set after epoch ends for a single validation function"""
        losses = []
        self.model.eval()  # Consider: since we can ask the state of the model
        # we can actuall use this to change the forward to track particular values
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                X, y = data
                self.to_device(X)
                self.to_device(y)

                y_hat = self.model.forward(X)
                loss = valid_loss_fn(y_hat, y)
                losses.append(loss)

        wandb.log({'validationloss': aggregate_fn(losses)}, step=self.step)

    def run(
            self,
            train_loader,
            test_loader,
            epochs,
            train_loss_fn,
            valid_loss_fns: Dict[str, Callable] = None,
            aggregate_fn: Callable = torch.mean
    ):
        """Main loop including training & test evaluation, all of which report to wandb"""

        if isinstance(train_loss_fn, DictConfig):
            train_loss_fn = instantiate(train_loss_fn)
        elif isinstance(train_loss_fn, Callable):
            pass

        if valid_loss_fns is not None:
            assert aggregate_fn is not None
            # self.losses = {k: [] for k in valid_loss_fns}

        for epoch in range(epochs):
            self.train(train_loader, epoch, train_loss_fn)

            if valid_loss_fns is not None:
                # End of epoch validation loss tracked in wandb
                for k, fn in valid_loss_fns.items():
                    loss = self.evaluate(test_loader, fn, aggregate_fn)
                    # self.losses[k].append(loss)
                    wandb.log({k: loss}, step=self.step)

            # execute other callbacks on the end of each epoch
            for callback in self.callbacks_end:
                callback(trainer=self)
