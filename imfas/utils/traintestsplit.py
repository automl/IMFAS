import random
from typing import List


def train_test_split(n, share):
    train_split = random.sample(list(range(n)), k=int(n * share))
    test_split = list(set(range(n)) - set(train_split))

    return train_split, test_split


def leave_one_out(n: int, idx: List[int]):
    """
    Leave one out cross validation
    This method allows to be explicit about which dataset is left out, when we want to run
    the experiments from command line. The idea is to run the experiment with a single holdout
    in one run with a fixed seed. Do n runs each with a different holdout and seed!
    """

    train_split = list(set(range(n)) - set(idx))
    test_split = idx

    return train_split, test_split
