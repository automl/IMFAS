import logging
import pathlib

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import DatasetMetaFeatures, AlgorithmMetaFeatures, Dataset_LC, \
    Dataset_Join
from mf_gravitas.util import seed_everything


# TODO debug flag to disable w&b & checkpointing.


@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    # todo seeding
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
    # fixme: move instantiation & join to lcbench.yaml
    algorithm_meta_features = AlgorithmMetaFeatures(
        path=dir_dataset_raw / 'config_subset.csv',
        transforms=instantiate(cfg.dataset.algo_meta_features),
        index_col=0
    )

    dataset_meta_features = DatasetMetaFeatures(
        path=dir_dataset_raw / 'meta_features.csv',
        transforms=instantiate(cfg.dataset.dataset_meta_features),
        index_col=0
    )

    lc_dataset = Dataset_LC(
        path=dir_dataset_raw / 'logs_subset.h5',
        transforms=instantiate(cfg.dataset.learning_curves),
        metric=cfg.dataset.lc_metric
    )

    # sync the datasets
    joint = Dataset_Join(
        dataset_meta_features,
        algorithm_meta_features,
        lc_dataset,
        competitors=2
    )

    # fixme: remove this loader: only for debugging purposes!
    joint_loader = torch.utils.data.DataLoader(
        joint,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=1
    )

    # for i, d in enumerate(joint_loader):
    #     print(i)
    #
    # # train test split by dataset major
    # train_split, test_split = train_test_split(len(dataset_meta_features), cfg.dataset.split)
    #
    # train_set = Dataset_Join_Split(
    #     meta_dataset=dataset_meta_features,
    #     meta_algo=algorithm_meta_features,
    #     lc=lc_dataset,
    #     splitindex=train_split,
    #     competitors=2,
    # )
    #
    # test_set = Dataset_Join_Split(
    #     meta_dataset=dataset_meta_features,
    #     meta_algo=algorithm_meta_features,
    #     lc=lc_dataset,
    #     splitindex=test_split,
    #     competitors=2,
    # )
    #
    # # Dataloaders
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=2
    # )
    #
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=cfg.batch_size,
    #     shuffle=True,
    #     num_workers=2
    # )

    # set hte number of algoritms and datasets
    cfg.model.model.input_dim = dataset_meta_features.df.columns.size
    cfg.model.model.n_algos = len(algorithm_meta_features)
    print(cfg.model.model)

    model = instantiate(cfg.model.model)

    model.train_gravity(
        joint_loader,
        joint_loader,
        epochs=[100, 100],
        lr=0.001
    )

    # select device

    # train model call

    # checkpoint model into output/date/time/ folder

    # evaluation model
    # test loader must be queried by the model.


if __name__ == '__main__':
    pipe_train()
