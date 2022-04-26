import logging
import pathlib

import hydra
import torch
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import Dataset_Join_Split
from mf_gravitas.util import seed_everything, train_test_split, measure_embedding_diversity
from mf_gravitas.evaluation.optimal_rankings import ZeroShotOptimalDistance
from torch.utils.data import DataLoader

import wandb
import os 
import pdb
import sys

import string
import random



def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

base_dir = os.getcwd()

@hydra.main(config_path='config', config_name='base')
def pipe_train(cfg: DictConfig) -> None:
    
    sys.path.append(os.getcwd())
    sys.path.append("..")

    dict_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)

    hydra_job = (
        os.path.basename(os.path.abspath(os.path.join(HydraConfig.get().run.dir, "..")))
        + "_"
        + os.path.basename(HydraConfig.get().run.dir)
    )

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
        main_raw(cfg.dataset_raw)

    # read in the data
    algorithm_meta_features = instantiate(cfg.dataset.algo_meta_features)
    dataset_meta_features = instantiate(cfg.dataset.dataset_meta_features)
    lc_dataset = instantiate(cfg.dataset.learning_curves)

    # train test split by dataset major
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
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )


    # update the input dims adn number of algos based on the sampled stuff
    cfg.model.model.input_dim = dataset_meta_features.df.columns.size
    cfg.model.model.n_algos = len(algorithm_meta_features)

    wandb.config.update({
        'n_algos': len(algorithm_meta_features),
        'input_dim': dataset_meta_features.df.columns.size
    })


    model = instantiate(cfg.model.model)

    #wandb.watch(model, log_freq=)
    
    call(
        cfg.training.schedule, 
        model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=cfg.training.schedule.epochs
    )



    # TODO checkpoint model into output/date/time/folder

    # evaluation model
    # # fixme: move this to config and instantiate
    evaluator = instantiate(
        cfg.evaluation.evaluator, 
        model=model,
        _recursive_=False 
    )
   
    counts = evaluator.forward(
        dataset_meta_features[test_split],
        final_performances=lc_dataset[test_split],
        steps=cfg.evaluation.steps)
    
    print(counts)


if __name__ == '__main__':
   pipe_train()