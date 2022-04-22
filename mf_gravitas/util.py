import os
import random
import warnings

import numpy as np
import torch
from networkx import Graph, minimum_spanning_edges
import pandas as pd


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


def calc_min_eucl_spanning_tree(d_test: torch.tensor):
    """
    calculates the minimum spanning tree of the euclidean distance matrix

    :param d_test: torch.tensor: the euclidean distance matrix

    :return: torch.tensor: the minimum spanning tree

    """
    dist_mat = torch.cdist(d_test, d_test)
    dist_mat = dist_mat.cpu().detach().numpy()

    nodes = list(range(len(dist_mat)))
    d = [(src, dst, dist_mat[src, dst]) for src in nodes for dst in nodes if src != dst]

    df = pd.DataFrame(data=d, columns=['src', 'dst', 'eucl'])

    g = Graph()
    for index, row in df.iterrows():
        g.add_edge(row['src'], row['dst'], weight=row['eucl'])

    return list(minimum_spanning_edges(g))  


def check_diversity(representation, title, epsilon=0.01):
    """
    :param representation: ndarray.
    :param title: name of the matrix
    :param epsilon: float: the value needed to exceed (should be close to zero)
    :raises: Warning if representation is not diverse
    """
    # Check for (naive) representation collapse by checking sparsity after
    # translation by 90% quantile
    translated = representation - np.quantile(representation, 0.9, axis=0)
    sparsity = (translated < epsilon).sum() / np.product(representation.shape)
    if sparsity >= 0.95:
        warnings.warn(f'The {title} representation is not diverse.')

        # Warning(f'The {title} representation is not diverse.')
        #print(representation)

def measure_embedding_diversity(model, data):
        """
        Calculate the diversity based on euclidiean minimal spanning tree
        :return:  diversity for datasets, diversity for algos
        """

        data_fwd = model.encode(data)
        z_algo = model.Z_algo

        data_tree = calc_min_eucl_spanning_tree(data_fwd)
        z_algo_tree = calc_min_eucl_spanning_tree(z_algo)

        d_diversity = sum([tup[2]['weight'] for tup in data_tree])
        z_diversity = sum([tup[2]['weight'] for tup in z_algo_tree])

        # sum of weighted edges
        return d_diversity, z_diversity