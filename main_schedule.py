import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from imfas.scheduler.environment import Environment

# A logger for this file
log = logging.getLogger(__name__)

import torch

OmegaConf.register_new_resolver("device_ident", lambda _: torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"))

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
OmegaConf.register_new_resolver("len", lambda l: len(l))

OmegaConf.register_new_resolver("range", lambda start, stop, step: list(range(start, stop, step)))

from main import pipe_train
import random
import string


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@hydra.main(config_path="configs", config_name="base_scheduler")
def pipe_scheduler(cfg: DictConfig) -> None:
    # optionally execute pipe_train to generate a transformer "interpreter"
    pipe_train(cfg)  # TODO: model must be saved to disk and loaded here

    # load saved model
    regret_model = instantiate(cfg.model)
    regret_model.load_state_dict(torch.load(cfg.model.model_path))

    env = Environment(cfg.environment, regret_model=regret_model)

    agent = instantiate(cfg.agent)
