import logging

from mf_gravitas.trainer.base_trainer import Trainer_Rank

import wandb


# A logger for this file
log = logging.getLogger(__name__)

def train_rank(model, train_dataloader, train_labels, test_dataloader, test_labels, epochs, lr ):
    trainer = Trainer_Rank()
    loss_fn = model.loss
    trainer.train(model, loss_fn, train_dataloader, train_labels, test_dataloader, test_labels, epochs, lr )

    

