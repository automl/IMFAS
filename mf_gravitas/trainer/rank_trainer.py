import logging

from mf_gravitas.trainer.base_trainer import Trainer_Rank

import wandb


# A logger for this file
log = logging.getLogger(__name__)

def train_rank(model, train_dataloader, test_dataloader, epochs, lr ):
    trainer = Trainer_Rank()
    loss_fn = model.loss
    score = trainer.train(model, loss_fn, train_dataloader, test_dataloader, epochs, lr )
    return score
    

