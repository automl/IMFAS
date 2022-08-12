import logging
import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)

import os
import random
import string
import sys
import getpass
from pathlib import Path
import shutil

import torch

import wandb
from hydra.utils import get_original_cwd
from tqdm import tqdm

from imfas.util import print_cfg, seed_everything, train_test_split

import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from imfas.losses.ranking_loss import spear_halve_loss


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


base_dir = os.getcwd()


@hydra.main(config_path="configs", config_name="base")
def pipe_train(cfg: DictConfig) -> None:
    sys.path.append(os.getcwd())
    sys.path.append("..")
    print("base_dir: ", base_dir)

    save_trained_models = True

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    print_cfg(cfg)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )

    hydra_config = HydraConfig.get()
    log.info(get_original_cwd())

    cfg.wandb.id = hydra_job + "_" + id_generator()

    run = wandb.init(**cfg.wandb, config=dict_cfg)

    hydra_cfg = HydraConfig.get()
    command = f"{hydra_cfg.job.name}.py " + " ".join(hydra_cfg.overrides.task)
    if not OmegaConf.is_missing(hydra_cfg.job, "id"):
        slurm_id = hydra_cfg.job.id
    else:
        slurm_id = None

    wandb.config.update({"command": command, "slurm_id": slurm_id})

    orig_cwd = hydra.utils.get_original_cwd()

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    seed_everything(cfg.seed)

    dir_data = pathlib.Path(cfg.dataset_raw.dir_data)
    dir_raw = dir_data / "raw"
    dir_dataset_raw = dir_data / "raw" / cfg.dataset_raw.dataset_name

    # optionally download / resubset the dataset
    if cfg.dataset_raw.enable:
        call(cfg.dataset_raw, _recursive_=False)

    dataset_meta_features = instantiate(cfg.dataset.dataset_meta)

    # train test split by dataset major
    train_split, test_split = train_test_split(
        len(dataset_meta_features),  # todo refactor - needs to be aware of dropped meta features
        cfg.dataset.split,
    )

    # Create the dataloaders
    train_set = instantiate(cfg.dataset.dataset_class, split=train_split)
    test_set = instantiate(cfg.dataset.dataset_class, split=test_split)

    train_loader = instantiate(cfg.dataset.dataloader_class, dataset=train_set, shuffle=True)
    test_loader = instantiate(cfg.dataset.dataloader_class, dataset=test_set)

    dataset_name = cfg.dataset.name
    model_name = cfg.model._target_.split('.')[-1]

    if save_trained_models:
        model_path = Path('/home') / getpass.getuser() / 'tmp' / 'IFMAS' / 'models' / \
                     dataset_name / model_name / cfg.training.loss_type / str(cfg.seed)

    # update the input dims and number of algos based on the sampled stuff
    # if "n_algos" not in cfg.dataset_raw.keys() and cfg.dataset.name != "LCBench":
    if not cfg.model._target_.split(".")[-1] == "HalvingGridSearchCV":
        n_meta_feature = dataset_meta_features.df.columns.size
        n_algos = len(train_set.lc.index)
        wandb.config.update({"n_algos": n_algos, "n_meta_feature": n_meta_feature})
        if cfg.model._target_ == 'imfas.models.hierarchical_transformer.HierarchicalTransformer':
            n_algo_features = train_set.meta_algo.transformed_df.shape[-1]
            model = instantiate(cfg.model,
                                input_dim_meta_feat=n_meta_feature,
                                input_dim_algo_feat=n_algo_features,
                                input_dim_lc=1)
        else:
            model = instantiate(cfg.model, input_dim=n_meta_feature, algo_dim=n_algos)

        valid_score = call(
            cfg.training,
            model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            _recursive_=False,
        )

        if save_trained_models:
            if not model_path.exists():
                os.makedirs(str(model_path), exist_ok=True)
            torch.save(model.state_dict(), str(model_path / 'model_weights.pt'))
        exit()

    else:

        if cfg.dataset.name == "LCBench":
            cfg.model.param_grid.algo_id = list(range(len(train_set.lc.index)))

        enable_halving_search_cv  # ensures import is not removed in alt + L reformatting

        # model.estimator.slices.split == test_split --this way datasets are parallel in seeds
        spears = {}
        for d in tqdm(test_split):
            # indexed with 0 and slices.split holds the relevant data id already!
            cfg.model.estimator.slices.split = [d]
            model: torch.nn.Module = instantiate(cfg.model, _convert_="partial")

            # fixme: validation score should not be computed during training!
            valid_score = call(
                cfg.training,
                model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                _recursive_=False,
            )

            final_performances = test_set.lc.transformed_df[d][-1]

            spears[d] = spear_halve_loss(valid_score, final_performances).numpy()

        # fixme: spearman is a constant for all test datasets.
        d = pd.DataFrame.from_dict(spears, orient="index")
        print(d)

        if cfg.dataset.name == "LCBench":
            name = "LCBench_raw"
        else:
            name = cfg.dataset_raw.bench

        d.to_csv(f"halving_test_spear_{name}_{cfg.seed}.csv")


if __name__ == "__main__":
    pipe_train()
