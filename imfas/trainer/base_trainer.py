from typing import Callable, List

import torch.optim
import wandb


def wandb_callback(trainer, ):
    # FIXME: @Aditya, if we wanted to have a running mean,
    #  why don't we let wandb let it do that for us?
    if trainer.step % trainer.log_freq == 0:
        for k, v in trainer.losses.items():
            wandb.log({k: v}, step=trainer.step)

        if trainer.step % trainer.log_freq == 0:

            for key in trainer.losses:
                trainer.losses[key] = torch.stack(trainer.losses[key]).mean()

            wandb.log(trainer.losses, commit=False, step=trainer.step)

            for key in trainer.losses:
                trainer.losses[key] = []


class BaseTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            callbacks_end: List[Callable]):

        self._step = 0  # FIXME: @Aditya: why exactly doesn't it suffice to have the epoch?
        self.model = model
        self.optimizer = optimizer
        self.callbacks_end = callbacks_end

    @property
    def step(self):
        return self._step

    def train(self, train_loader, epochs, lr, loss_fn):
        """define one epoch of training"""
        # TODO: incorporate callbacks to the training loop (before and after each epoch, and )
        #  at every (k-th) step?
        for _, data in enumerate(train_loader):
            self.optimizer.zero_grad()

            # fixme: move data to device (optionally directly on the dataset iterator)?
            #  because then we can go shape agnostic!
            X = data[0].to(self.model.device)
            y = data[1].to(self.model.device)

            y_hat = self.model.forward(data)
            loss = loss_fn(y_hat, y).backward()
            self.optimizer.step()

    def evaluate(self, test_loader, valid_loss_fn, aggregate_fn):
        """evaluate the model on the test set after epoch ends"""
        losses = []
        self.model.eval()
        with torch.no_grad():
            for _, data in enumerate(test_loader):
                # fixme: move data to device
                X = data[0].to(self.model.device)
                y = data[1].to(self.model.device)

                y_hat = self.model.forward(X)
                loss = valid_loss_fn(y_hat, y)
                losses.append(loss)

        return aggregate_fn(losses)

    def run(self, train_loader, test_loader, epochs, lr, train_loss_fn,
            valid_loss_fns: List[Callable], aggregate_fn: Callable):
        """Main loop including training & test evaluation"""
        self.losses = {k: [] for k in valid_loss_fns}
        for epoch in range(epochs):
            self.train(train_loader, epoch, lr, train_loss_fn)

            # evaluate on all losses
            for k, valid_loss_fn in valid_loss_fns.items():
                self.losses[k].append(self.evaluate(test_loader, valid_loss_fn, aggregate_fn))

            self._step += 1

            for callback in self.callbacks_end:
                callback(self)
