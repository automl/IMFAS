import logging
import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.util import seed_everything, train_test_split, print_cfg

import wandb
import os
import sys

import string
import random

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
        # FIXME: check if anything (LHD, nalgos, dataset slices, ...) changed,
        #   and trigger a recalculation automatically - so we need to write out the
        #   config alongside the dataset, that generated the dataset - and compare
        #   the current against it.
        call(cfg.dataset_raw, _recursive_=False)

        # log.info(f'\n{"!" * 30}\nTerminating after generating raw data. To continue, override your '
        #          'config with "dataset_raw.enable=false" ')
        #
        # return None

    # move this definition into the config file of dataset_join_dmajor
    # read in the data
    # algorithm_meta_features = instantiate(cfg.dataset.algo_meta)
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta)  # fixme this is still required
    # lc_dataset = instantiate(cfg.dataset.lc_meta)

    # train test split by dataset major
    train_split, test_split = train_test_split(
        len(dataset_meta_features),  # todo refactor - needs to be aware of dropped meta features
        cfg.dataset.split
    )

    # dataset_major
    # fixme: refactor this into a configurable class! - either dmajor or multidex (the latter for
    #  algo meta features & dataset

    train_set = instantiate(cfg.dataset.dataset_class, split=train_split)
    test_set = instantiate(cfg.dataset.dataset_class, split=test_split)

    train_loader = instantiate(cfg.dataset.dataloader_class, dataset=train_set)
    test_loader = instantiate(cfg.dataset.dataloader_class, dataset=test_set)

    # update the input dims adn number of algos based on the sampled stuff
    if 'n_algos' not in cfg.dataset_raw.keys():  # todo refactor this if statement
        input_dim = len(train_set.meta_dataset.df.index)
        n_algos = len(train_set.lc.index)  # fixme: instead calculate from joint dataset or
        # directly in config! (number of algorithms! careful with train/test split!)

        wandb.config.update({
            'n_algos': n_algos,
            'input_dim': input_dim
        })

        model = instantiate(
            cfg.model,
            input_dim=input_dim,
            algo_dim=n_algos
        )

        valid_score = call(
            cfg.training,
            model,
            train_dataloader=train_loader,
            test_dataloader=test_loader,
            _recursive_=False
        )

    elif cfg.model._target_.split('.')[-1] == 'HalvingGridSearchCV':
        # fixme: refactor this if
        #  branch, that is specificaly targeting the HalvingGridSearchCV.

        # explicitly required since it is an experimental feature

        from mf_gravitas.losses.ranking_loss import spear_halve_loss
        from sklearn.experimental import enable_halving_search_cv
        import pandas as pd
        enable_halving_search_cv  # ensures import is not removed in alt + L reformatting

        # model.estimator.slices.split == test_split --this way datasets are parallel in seeds
        spears = {}
        for d in test_split:
            # indexed with 0 and slices.split holds the relevant data id already!
            cfg.model.estimator.slices.split = [d]
            model = instantiate(cfg.model, _convert_='partial')

            # fixme: validation score should not be computed during training!
            valid_score = call(
                cfg.training,
                model,
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                _recursive_=False
            )

            final_performances = test_set.lc.transformed_df[d][-1]

            spears[d] = spear_halve_loss(valid_score, final_performances).numpy()

        # fixme: spearman is a constant for all test datasets.
        pd.DataFrame.from_dict(spears, orient='index').to_csv('halving_test_spear.csv')

    # TODO trainsize config incl rescaled 100
    return valid_score  # needed for smac


if __name__ == '__main__':
    pipe_train()
