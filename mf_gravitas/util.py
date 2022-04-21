import os
import random

import numpy as np
import torch

import numpy as onp
import pandas as pd
import wandb



def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_test_split(n, share):
    train_split = random.sample(
        list(range(n)),
        k=int(n * share)
    )

    test_split = list(set(range(n)) - set(train_split))

    return train_split, test_split


# TODO wandb logging
'''
    - Loss
    - validation score
    - MST representation
'''
def log_wandb(model: torch.nn.Module):
    metrics = {
        "train/episode": train_monitor_env.ep,
        "train/avg_reward": train_monitor_env.avg_r,
        "train/return": train_monitor_env.G,
        "train/steps": train_monitor_env.t,
        "train/avg_step_duration_ms": train_monitor_env.dt_ms,
    }

    wandb.log(metrics, step=train_monitor_env.T)
