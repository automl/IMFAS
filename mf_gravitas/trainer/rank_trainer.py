# fixme: move this entire file's wandb logging into the trainer classes, where they'd belong

import logging

import torch
import torchsort
import wandb
from tqdm import tqdm

from mf_gravitas.losses.ranking_loss import spearman
from mf_gravitas.trainer.rank_ensemble import Trainer_Ensemble
from mf_gravitas.trainer.rank_trainer_class import Trainer_Rank

# A logger for this file
log = logging.getLogger(__name__)


def train_rank(model, train_dataloader, test_dataloader, epochs, lr):
    trainer = Trainer_Rank()
    loss_fn = model.loss
    slice_index = -1

    print(train_dataloader.dataset.slice_indices)

    kwargs = {
        'model': model,
        'loss_fn': loss_fn,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'epochs': epochs,
        'lr': lr,
        'slice_index': slice_index,
    }

    score, step = trainer.train(**kwargs)

    wandb.log(
        score,
        commit=False,
        step=step
    )

    return score


def train_ensemble(model, train_dataloader, test_dataloader, epochs, lr,
                   ranking_fn=torchsort.soft_rank, optimizer_cls=torch.optim.Adam):
    """
    
    """

    optimizer = optimizer_cls(
        model.parameters(),
        lr
    )

    trainer_kwargs = {
        'model': model,
        'loss_fn': spearman,
        'ranking_fn': ranking_fn,
        'optimizer': optimizer,
    }

    # Initialize the trainer
    trainer = Trainer_Ensemble(**trainer_kwargs)

    for e in tqdm(range(epochs)):
        # Train the model
        trainer.train(train_dataloader)

        # Evaluate the model
        score = trainer.evaluate(test_dataloader)

        # Take the next step
        trainer.step_next()

        wandb.log(
            trainer.losses,
            commit=False,
            step=e
        )

    return score


def train_ensemble_freeze(model, train_dataloader, test_dataloader, lr, epochs=[300, 500],
                          ranking_fn=torchsort.soft_rank, optimizer_cls=torch.optim.Adam):
    """
    Freezing the final joint model to foster stable learning in the multihead
    fidelities. (To avoid the double gradient on them earlier components in the
    earliy stages of training. This supposedly gets us a decent initialization)
    """

    optimizer = optimizer_cls(
        model.parameters(),  # fixme: freeze some parameters
        lr
    )
    trainer_kwargs = {
        'model': model,
        'loss_fn': spearman,
        'ranking_fn': ranking_fn,
        'optimizer': optimizer,
    }

    # Initialize the trainer
    trainer = Trainer_Ensemble(**trainer_kwargs)

    log.info('Starting Freezed pretraining')
    for e in tqdm(range(epochs[0])):
        # Train the model
        trainer.train(train_dataloader)

        # Evaluate the model
        score = trainer.evaluate(test_dataloader)

        # Take the next step
        trainer.step_next()

        wandb.log(
            trainer.losses,
            commit=False,
            step=e
        )

    log.info('Training fully')
    for e in tqdm(range(epochs[1])):
        # Train the model
        trainer.train(train_dataloader)

        # Evaluate the model
        score = trainer.evaluate(test_dataloader)

        # Take the next step
        trainer.step_next()

        wandb.log(
            trainer.losses,
            commit=False,
            step=e + epochs[0]
        )

    return score
