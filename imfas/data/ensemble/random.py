# TODO: add random ensemble
import numpy as np


def random_ensemble(n: int, df, **kwargs):
    """
    Randomly select k algorithms from the set of all algorithms.
    :param df: of final performances of all algorithms on all datasets. (not relevant to
    the ensemble selection procedure except, that we can have the names of the algorithms)
    #--> fixme: change the interface of the rawpipe
    :param n:
    :return: tuple[set, pd.Dataframe] set of candidates, the respective performance profiles.
    """
    algos = df.index
    candidates = set(np.random.choice(algos, n, replace=False))

    return candidates, df.loc[candidates]
