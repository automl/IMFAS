# fixme: move this entire file's wandb logging into the trainer classes, where they'd belong

import logging

import torch
import torchsort
import wandb
from tqdm import tqdm

from mf_gravitas.losses.ranking_loss import spearman, weighted_spearman
from mf_gravitas.trainer.rank_trainer_class import Trainer_Rank
from mf_gravitas.trainer.lstm_trainer import Trainer_Ensemble_lstm


# A logger for this file
log = logging.getLogger(__name__)

def train_lstm(model, train_dataloader, test_dataloader, epochs, lr,
                   ranking_fn=torchsort.soft_rank, optimizer_cls=torch.optim.Adam, test_lim=5, log_freq=10):
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
        'test_lim': test_lim
    }

    # Initialize the trainer
    trainer = Trainer_Ensemble_lstm(**trainer_kwargs)

    losses = {}
    
    init  = True

    for e in tqdm(range(epochs)):
        # Train the model
        trainer.train(train_dataloader)

       

        # Evaluate the model
        trainer.evaluate(test_dataloader)

        if init:
            for key in trainer.losses:
                losses[key] = []
            
            init = False
        
        # Take the next step
        trainer.step_next()

        for key in trainer.losses:
            losses[key].append(trainer.losses[key])
        

        if e%log_freq==0:
            
            for key in trainer.losses:
                losses[key]= torch.stack(losses[key]).mean()
            
            
            wandb.log(
                losses,
                commit=False,
                step=e
            )

            for key in trainer.losses:
                losses[key] = []


    # return score
