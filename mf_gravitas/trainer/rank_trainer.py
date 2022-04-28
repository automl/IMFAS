import logging

from mf_gravitas.trainer.base_trainer import Trainer_Ensemble, Trainer_Rank

import wandb

import pdb

# A logger for this file
log = logging.getLogger(__name__)

def train_rank(model, train_dataloader, test_dataloader, epochs, lr ):
    trainer = Trainer_Rank()
    loss_fn = model.loss
    slice_index = -1

    print(train_dataloader.dataset.slice_indices)
    pdb.set_trace()

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
    

def train_ensemble(model, train_dataloader, test_dataloader, epochs, lr):
    """
    
    """
    trainer = Trainer_Ensemble()
    loss_fn = model.loss
    

    kwargs = {
        'model': model,
        'loss_fn': loss_fn,
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'epochs': epochs,
        'lr': lr,
    }

    score, step = trainer.train(**kwargs)

    wandb.log(
        score,
        commit=False,
        step=step
    )

    return score

    
   

    

