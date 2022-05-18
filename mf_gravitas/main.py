import logging
import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data import Dataset_Join_Dmajor
from mf_gravitas.util import seed_everything, train_test_split, print_cfg
from torch.utils.data import DataLoader

import wandb
import os
import sys

import string
import random
import torch

from hydra.utils import get_original_cwd


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


base_dir = os.getcwd()


@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    sys.path.append(os.getcwd())
    sys.path.append("..")
    print('base_dir: ', base_dir)

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    print_cfg(cfg)

    hydra_job = (
            os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
            + "_"
            + os.path.basename(HydraConfig.get().run.dir)
    )

    # fixme: remove temporary quickfix to change pass smac config into the hydra config
    hydra_config = HydraConfig.get()
    log.info(get_original_cwd())

    cfg.wandb.id = hydra_job + "_" + id_generator()

    run = wandb.init(
        **cfg.wandb,
        config=dict_cfg
    )

    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None

    wandb.config.update({
        "command": command,
        "slurm_id": slurm_id
    })

    orig_cwd = hydra.utils.get_original_cwd()

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    seed_everything(cfg.seed)

    dir_data = pathlib.Path(cfg.dataset_raw.dir_data)
    dir_raw = dir_data / 'raw'
    dir_dataset_raw = dir_data / 'raw' / cfg.dataset_raw.dataset_name

    # optionally download / resubset the dataset
    if cfg.dataset_raw.enable:
        call(cfg.dataset_raw, _recursive_=False)

        log.info(f'\n{"!" * 30}\nTerminating after generating raw data. To continue override your '
                 'config with "dataset_raw.enable=false" ')

        return None

    # read in the data
    algorithm_meta_features = instantiate(cfg.dataset.algo_meta)
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta)
    lc_dataset = instantiate(cfg.dataset.lc)

    # train test split by dataset major
    train_split, test_split = train_test_split(
        len(dataset_meta_features),
        cfg.dataset.split
    )

    # dataset_major
    # fixme: refactor this into a configurable class! - either dmajor or multidex (the latter for
    #  algo meta features & dataset
    train_set = Dataset_Join_Dmajor(
        meta_dataset=dataset_meta_features,
        lc=lc_dataset,
        split=train_split
    )

    test_set = Dataset_Join_Dmajor(
        meta_dataset=dataset_meta_features,
        lc=lc_dataset,
        split=test_split
    )

    # wrap with Dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train_batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # update the input dims adn number of algos based on the sampled stuff
    input_dim = dataset_meta_features.df.columns.size
    n_algos = len(algorithm_meta_features)

    wandb.config.update({
        'n_algos': n_algos,
        'input_dim': input_dim
    })

    model = instantiate(
        cfg.model,
        input_dim=input_dim,
        algo_dim=n_algos
    )

    # fixme: validation score should not be computed during run !
    valid_score = call(
        cfg.training,
        model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        _recursive_=False
    )

    # fixme: remove highly specific weighing scheme for multihead losses!

    head_losses = torch.tensor(valid_score)
    fidelity_weights = torch.nn.functional.softmax(torch.log(
        torch.tensor(cfg.dataset.slices[:-1],  # multihead does not need the last fidelity!
                     dtype=torch.float)))
    valid_score = fidelity_weights @ head_losses.type(torch.float).T

    return valid_score


if __name__ == '__main__':
    pipe_train()
