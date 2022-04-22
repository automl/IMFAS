import logging
import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import Dataset_Join_Split
from mf_gravitas.util import seed_everything, train_test_split
from torch.utils.data import DataLoader

import wandb
import os

# TODO debug flag to disable w&b & checkpointing.

base_dir = os.getcwd()


@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    hydra_cfg = HydraConfig.get()
    # print(hydra_cfg)

    log.info("W&B was initialized")
    wandb.init(**cfg.wandb)

    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    seed_everything(cfg.seed)

    # FIXME move data_dir to cfg!
    orig_cwd = hydra.utils.get_original_cwd()
    dir_data = pathlib.Path(cfg.dataset_raw.dir_data)
    dir_raw = dir_data / 'raw'
    dir_dataset_raw = dir_data / 'raw' / cfg.dataset_raw.dataset_name
    # init w&b and convert config_raw for w&b

    # optionally download / resubset the dataset
    if cfg.dataset_raw.enable:
        log.info('Download/Recompute subsets from raw.')
        main_raw(cfg.dataset_raw)

    # read in the data
    log.info('Instantiating partial datasets')
    algorithm_meta_features = instantiate(cfg.dataset.algo_meta_features)
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta_features)
    lc_dataset = instantiate(cfg.dataset.learning_curves)

    # # train test split by dataset major
    log.info('Train-test-split & merging of partial datasets into joint dataloaders')
    train_split, test_split = train_test_split(len(dataset_meta_features), cfg.dataset.split)

    train_set = Dataset_Join_Split(
        meta_dataset=dataset_meta_features,
        meta_algo=algorithm_meta_features,
        lc=lc_dataset,
        splitindex=train_split,
        competitors=cfg.num_competitors,
    )

    test_set = Dataset_Join_Split(
        meta_dataset=dataset_meta_features,
        meta_algo=algorithm_meta_features,
        lc=lc_dataset,
        splitindex=test_split,
        competitors=cfg.num_competitors,
    )

    # Dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    log.info('Updating w&b/hydra config of (dynamically adjusted) input dimensions.')
    # FIXME: make this available to w&b after the update!
    cfg.model.model.input_dim = dataset_meta_features.df.columns.size
    cfg.model.model.n_algos = len(algorithm_meta_features)
    print(cfg.model.model)

    log.info('instantiate model')
    model = instantiate(cfg.model.model)

    wandb.watch(model, log_freq=1)

    log.info(f'Training model with {cfg.training.schedule}')
    call(
        cfg.training.schedule,
        model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=cfg.training.schedule.epochs
    )

    # TODO checkpoint model into output/date/time/folder

    # evaluation model
    # fixme: move this to config and instantiate
    log.debug(f'Evaluating model with {cfg.evaluation.evaluator}')
    evaluator = instantiate(
        cfg.evaluation.evaluator,
        model=model,
        _recursive_=False
    )

    evaluator.forward(
        dataset_meta_features[test_split],
        final_performances=lc_dataset[test_split],
        steps=cfg.evaluation.steps)


if __name__ == '__main__':
    pipe_train()
