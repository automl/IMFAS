import logging
import wandb

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch

# A logger for this file
log = logging.getLogger(__name__)

# TODO debug flag to disable w&b & checkpointing.


@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    # init w&b and convert config for w&b

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config")
    # log.debug("Debug level message")

    # seeding

    # load dataset

    # create dataloader from it

    # train test split

    # instantiate model
    model = instantiate(cfg.model)

    # select device

    # train model

    # checkpoint model into output/date/time/ folder

    # evaluation model



if __name__ == '__main__':
    pipe_train()


