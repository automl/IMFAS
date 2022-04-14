import os
import random

import numpy as np
import torch


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
