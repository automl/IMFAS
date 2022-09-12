import logging

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)

import torch

OmegaConf.register_new_resolver(
    "device_ident", lambda _: torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

OmegaConf.register_new_resolver(
    "add", lambda *numbers: sum(numbers)
)
OmegaConf.register_new_resolver(
    "len", lambda l: len(l)
)

OmegaConf.register_new_resolver(
    "range", lambda start, stop, step: list(range(start, stop, step))
)

import os
import random
import string

import wandb
from hydra.utils import get_original_cwd

from imfas.util import print_cfg, seed_everything, train_test_split


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@hydra.main(config_path="configs", config_name="base")
def pipe_train(cfg: DictConfig) -> None:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    print_cfg(cfg)

    log.info(get_original_cwd())

    # FIXME: W&B id???
    hydra_job = (
            os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
            + "_"
            + os.path.basename(HydraConfig.get().run.dir)
    )
    cfg.wandb.id = hydra_job + "_" + id_generator()  # fixme: necessary?

    run = wandb.init(**cfg.wandb, config=dict_cfg)

    # FIXME: SLURM ID???
    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None

    wandb.config.update({"command": command, "slurm_id": slurm_id})

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))  # FIXME: pretty print???!!!

    seed_everything(cfg.seed)

    # Create / find + setup the data -------------------------------------------
    # optionally download / resubset the dataset
    if cfg.dataset.dataset_raw.enable:
        # fixme: rather than enable: save a meta info string of the dataset_raw config,
        #  that generated the data and check if it is consistent; else trigger recompute!
        call(cfg.dataset.dataset_raw, _recursive_=False)

    # dataset_meta_features must be instantiated to give a traintest split index
    # of the data, that can be passed on.
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta)

    # train test split by dataset major
    train_split, test_split = train_test_split(
        len(dataset_meta_features),  # fixme refactor - needs to be aware of dropped meta features
        cfg.dataset.split,
    )

    # Create the dataloaders
    train_set = instantiate(cfg.dataset.train_dataset_class, split=train_split)
    test_set = instantiate(cfg.dataset.test_dataset_class, split=test_split)

    train_loader = instantiate(cfg.dataset.test_dataloader_class, dataset=train_set)
    test_loader = instantiate(cfg.dataset.test_dataloader_class, dataset=test_set)

    # Dynamically computed configurations.
    # maybe change later to resolvers? https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#access-and-manipulation
    cfg.dynamically_computed.n_data_meta_features = dataset_meta_features.df.columns.size
    cfg.dynamically_computed.n_algos = len(train_set.lc)
    cfg.dynamically_computed.n_algo_meta_features = train_set.meta_algo.transformed_df.shape[-1]

    wandb.config.update(
        {"dynamically_computed.n_algos": train_set.meta_algo.transformed_df.shape[-1],
         "dynamically_computed.n_data_meta_features": dataset_meta_features.df.columns.size}
    )

    # CLASSICAL MODELS -----------------------------------------------------------------------------
    model = instantiate(cfg.model)
    model.to(cfg.device)

    trainer = instantiate(cfg.trainer.trainerobj, model)
    trainer.run(
        train_loader=train_loader,
        test_loader=test_loader,
        **cfg.trainer.run_call,
    )

    # FIXME: move this to the trainer as a model.save call!
    # if cfg.save_models:
    #     model_path = Path('/home') / getpass.getuser() / 'tmp' / 'IFMAS' / 'models' / \
    #                  dataset_name / model_name / cfg.training.loss_type / str(cfg.seed)
    #     if not model_path.exists():
    #         os.makedirs(str(model_path), exist_ok=True)
    #     torch.save(model.state_dict(), str(model_path / 'model_weights.pt'))
    # exit()


if __name__ == "__main__":
    pipe_train()
