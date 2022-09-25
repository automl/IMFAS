from typing import Union, Callable, Dict

from hydra.utils import instantiate
from omegaconf import DictConfig
from optuna.integration import wandb
from tqdm import tqdm

from imfas.trainer import BaseTrainer


class ASTrainer(BaseTrainer):
    def __init__(self, model):

        self.model = model

    def run(self, train_loader, valid_loader, test_loader,
            train_loss_fn: Union[Callable, DictConfig] = None,
            test_loss_fns: Dict[str, Callable] = None):

        self.model.train()
        X, y = next(train_loader)

        # fixme: how to make use of trainloss?
        self.model.forward(**X, **y)
        # Deliberately leave out the valid_loss loader (to make models comparable)

        self.model.eval()
        X, y = next(test_loader)
        y_hat = self.model.forward(**X)

        for fn_name, fn in tqdm(test_loss_fns.items(), desc='Test functions'):
            if isinstance(fn, DictConfig):
                fn = instantiate(fn)
                loss = fn(y_hat, y)

                wandb.log({f"Test: {fn_name}": loss.mean()})

# TODO transformer in basetrainer.
# TODO SH baseline in basetrainer (config)
# TODO AS baseline in basetrainer (check how)
# TODO LCDB with test curves.
