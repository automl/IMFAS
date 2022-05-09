import logging
import pathlib

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, call
from omegaconf import DictConfig, OmegaConf

import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

# A logger for this file
log = logging.getLogger(__name__)

from mf_gravitas.data.pipe_raw import main_raw
from mf_gravitas.data import Dataset_Join_Dmajor
from mf_gravitas.util import seed_everything, train_test_split
from torch.utils.data import DataLoader


import numpy as np

import wandb
import os
import sys

import string
import random

from mf_gravitas.trainer.base_trainer import Trainer_Rank
from mf_gravitas.models.rank_mlp import ActionRankMLP

from mf_gravitas.trainer.rank_trainer import train_rank


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


base_dir = os.getcwd()




def smac_train(cfg, train_dataloader, test_dataloader, input_dims, output_dims ):
    lr = cfg["learning_rate"]

    model = ActionRankMLP(
        input_dim = input_dims,
        hidden_dims=[cfg["n_neurons"]] * cfg["n_layer"],
        action_dim = output_dims,
    )


    # train model
    score = train_rank(model, train_dataloader, test_dataloader, cfg["epochs"], lr)

    return score

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
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers
    )

    # update the input dims adn number of algos based on the sampled stuff
    input_dim = dataset_meta_features.df.columns.size
    action_dim = len(algorithm_meta_features)

    wandb.config.update({
        'n_algos': len(algorithm_meta_features),
        'input_dim': dataset_meta_features.df.columns.size
    })

    #model = instantiate(cfg.model.model)
    n_layer = UniformIntegerHyperparameter("n_layer", 1, 5, default_value=1)
    n_neurons = UniformIntegerHyperparameter(
        "n_neurons", 8, 1024, log=True, default_value=10
    )
    learning_rate = CategoricalHyperparameter(
        "learning_rate",
        ["constant", "invscaling", "adaptive"],
        default_value="constant",
    )
    learning_rate_init = UniformFloatHyperparameter(
        "learning_rate_init", 0.0001, 1.0, default_value=0.001, log=True
    )

    cs = ConfigurationSpace()
    cs.add_hyperparameters(
        [
            n_layer,
            n_neurons,
            learning_rate,
            learning_rate_init,
        ]
    )

     # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": 100,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": True,
            # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            "limit_resources": False,
            "cutoff": 30,  # runtime limit for target algorithm
            "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 50

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3}

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4MF(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=smac_train,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    def_value = tae.run(
        config=cs.get_default_configuration(),
        budget=max_epochs,
        seed=123456,
        

    )[1]

    print("Value for default configuration: %.4f" % def_value)



if __name__ == '__main__':
    pipe_train()
