import logging
import pathlib

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import Dataset_Join_Split
from mf_gravitas.util import seed_everything, train_test_split
from mf_gravitas.evaluation.optimal_rankings import ZeroShotOptimalDistance
from torch.utils.data.dataloader import DataLoader

import wandb
import os 

import pdb

base_dir = os.getcwd()

@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    hydra_cfg = HydraConfig.get()
    # print(hydra_cfg)
    

    wandb.init(
        mode="offline" if cfg.debug else None,
        project="gravitas",
        entity="tnt",
        group=cfg.group,
        dir=os.getcwd(),
        config=dict_cfg,
    )

    
    orig_cwd = hydra.utils.get_original_cwd()

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    seed_everything(cfg.seed)

    # fixme: move data_dir to cfg!
    dir_data = pathlib.Path(cfg.dataset_raw.dir_data)
    dir_raw = dir_data / 'raw'
    dir_dataset_raw = dir_data / 'raw' / cfg.dataset_raw.dataset_name
    # init w&b and convert config_raw for w&b

    # optionally download / resubset the dataset
    if cfg.dataset_raw.enable:
        main_raw(cfg.dataset_raw)

    # read in the data
    algorithm_meta_features = instantiate(cfg.dataset.algo_meta_features)
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta_features)
    lc_dataset = instantiate(cfg.dataset.learning_curves)

    # # train test split by dataset major
    train_split, test_split = train_test_split(len(dataset_meta_features), cfg.dataset.split)

    train_set = Dataset_Join_Split(
        meta_dataset=dataset_meta_features,
        meta_algo=algorithm_meta_features,
        lc=lc_dataset,
        splitindex=train_split,
        competitors=2,
    )

    test_set = Dataset_Join_Split(
        meta_dataset=dataset_meta_features,
        meta_algo=algorithm_meta_features,
        lc=lc_dataset,
        splitindex=test_split,
        competitors=2,
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2
    )

    # set the number of algoritms and datasets
    # to make the number of datasets_meta features & n_algos dependent on the
    # used preprocessing, we need to update the config
    # FIXME: make this available to w&b after the update!
    cfg.model.model.input_dim = dataset_meta_features.df.columns.size
    cfg.model.model.n_algos = len(algorithm_meta_features)
    print(cfg.model.model)

    model = instantiate(cfg.model.model)

    wandb.watch(model)
    
    # todo select device
    model.train_schedule(
        train_loader,
        test_loader,
        epochs=[5, 5, 5],
        lr=0.001
    )

    
    # model.train_gravity(
    #     train_loader,
    #     test_loader,
    #     epochs=[1, 1],
    #     lr=0.001
    # )

    # TODO checkpoint model into output/date/time/ folder

    # evaluation model
    # fixme: move this to config and instantiate
    evaluator = ZeroShotOptimalDistance(
        model,
        ranking_loss=cfg.evaluation.ranking_loss
    )
    counts =  evaluator.forward(
        dataset_meta_features[test_split],
        final_performances=lc_dataset[test_split],
        steps=cfg.evaluation.steps)

    print(counts)


if __name__ == '__main__':
   pipe_train()