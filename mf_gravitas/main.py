import logging
import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import DatasetMetaFeatures, AlgorithmMetaFeatures


# TODO debug flag to disable w&b & checkpointing.


@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    orig_cwd = hydra.utils.get_original_cwd()

    # fixme: move data_dir to cfg!
    dir_data = pathlib.Path(cfg.dataset_raw.dir_data)
    dir_raw = dir_data / 'raw'
    dir_dataset_raw = dir_data / 'raw' / cfg.dataset_raw.dataset_name
    # init w&b and convert config_raw for w&b

    # optionally download /resubset the dataset
    main_raw(cfg.dataset_raw)

    algorithm_meta_features = AlgorithmMetaFeatures(
        path=dir_dataset_raw / 'config.csv',
        transforms=instantiate(cfg.dataset.algo_meta_features),
        index_col=0
    )

    dataset_meta_features = DatasetMetaFeatures(
        path=dir_dataset_raw / 'meta_features.csv',
        transforms=instantiate(cfg.dataset.dataset_meta_features),
        index_col=0
    )

    # logging TODO add logging to each step of the way.
    log.info("Hydra initialized a new config_raw")
    log.debug(str(cfg))

    # seeding

    # instantiate & preprocess meta

    # Fixme: write a joint Dataset class interface that allows batching, that makes each and every
    #  of these optional!

    # meta train

    # train test split

    # load dataset

    # create dataloader from it

    # instantiate model
    # model = instantiate(cfg.model)

    # select device

    # train model call

    # checkpoint model into output/date/time/ folder

    # evaluation model
    # test loader must be queried by the model.


if __name__ == '__main__':
    pipe_train()
