from typing import Optional
from random import randint

import numpy as np
import torch
from torch.utils.data import Dataset

from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC
from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures


# FIXME: Refactor joint - gravity, multiindex & compettitor sampling
class Dataset_Join_Gravity(Dataset):
    def __init__(self, meta_dataset, meta_algo, lc, competitors: int = 0):
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo
        self.lc = lc

        self.competitors = competitors

        # FIXME: add consistency checks on rows & columns!

        # FIXME: this is brittle and depends on the lc.transformed_df format after the pipe!
        # it also assumes that meta_dataset & meta_algo have the exact same ordering
        self.dataset_names, self.algo_names = self.lc.columns, self.lc.index

        # LC Style
        # self.multidex = deepcopy(lc.multidex)
        # self.multidex = self.multidex.set_levels([
        #     list(range(len(self.dataset_names))),
        #     list(range(len(self.algo_names)))
        # ])
        #

        # LC Slice Style
        # This index is useful with
        self.multidex = list(
            (d, a)
            for d in range(len(self.meta_dataset.transformed_df))
            for a in range(len(self.meta_algo.transformed_df))
        )

        # be aware of row columns in:
        # self.lc.df[51].unstack().T

    def __getitem__(self, item):
        """
        sync getitem across the multiple dataset classes.
        """
        D_m, A_m, a_p = self.__get_single__(item)
        if self.competitors > 0:
            competitors = self.__get_competitors__(item)

            return (D_m, a_p), competitors  # FIXME: algo meta features are disabled
        else:
            return (D_m, a_p), (None, None)

    def __get_single__(self, item):
        d, a = self.multidex[item]
        # FIXME: indexing depends on the transformations applied
        #  in particularly troubling is lc, since it is a time slice!
        return self.meta_dataset[d], self.meta_algo[a], self.lc[a]

    def __get_multiple__(self, items):
        """
        Fetch multiple items at once (output is like single, but with
        stacked tensors)
        """
        # parse the index & allow it to fetch multiple vectors at once
        # LC Slice style
        l = [self.multidex[i - 1] for i in items]
        d, a = zip(*l)

        # Consider using this when moving from LC Slice to LC
        # LC Style
        # d, a = zip(*self.multidex[items])

        d, a = list(d), list(a)
        # FIXME: indexing depends on the transformations applied
        #  in particularly troubling is lc, since it is a time slice!
        return self.meta_dataset[d], self.lc[d]  # self.meta_algo[a], # FIXME add in algo meta

    def __get_competitors__(self, item):
        # Consider: Creating the competitor set might be the bottleneck
        competitors = [randint(0, self.__len__()) for c in range(self.competitors)]
        competitors = [c for c in competitors if c != item]

        # ensure, we will never hit the same item as competitor
        if len(competitors) != self.competitors:
            while len(competitors) != self.competitors:
                val = randint(0, self.__len__())
                if val != item:
                    competitors.append(randint(0, self.__len__()))

        return self.__get_multiple__(competitors)

    def __len__(self):
        return len(self.meta_dataset.transformed_df) * len(self.meta_algo.transformed_df)


class Dataset_Join_Split(Dataset_Join_Gravity):
    def __init__(self, splitindex: list[int], *args, **kwargs):
        """
        Convenience wrapper around Dataset_Join_Gravity to
        Deterministically split it into train and test sets based on splitindex.
        :param splitindex: index of the datasets that are to be kept
        """
        super(Dataset_Join_Split, self).__init__(*args, **kwargs)
        self.splitindex = splitindex

        # Consider using this, when switching LC slice or LC!
        # self.multidex = pd.MultiIndex.from_tuples(
        #     [(d, a) for d, a in self.multidex if d in splitindex],
        #     names=['dataset', 'algorithm'])

        # This index is useful with
        self.multidex = list(
            (d, a) for d in self.splitindex for a in range(len(self.meta_dataset))
        )  # range(len(self.meta_algo.transformed_df)))

    def __len__(self):
        return len(self.splitindex) * len(self.meta_dataset)


class Dataset_Join_Dmajor(Dataset):
    def __init__(self, meta_dataset: DatasetMetaFeatures, lc: Dataset_LC,
                 meta_algo: Optional[AlgorithmMetaFeatures] = None, split=None):
        """
        joint major dataset?
        Args:
            meta_dataset: meta features with size [n_datasets, n_features]
            lc: lc bench dataset with size [n_dataset, n_slices, n_features]
            meta_algo:
            split:
        """
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo  # FIXME not required yet
        self.lc = lc

        if split is not None:
            self.split = split
        else:
            self.split = list(range(len(self.meta_dataset)))

    def __getitem__(self, item):
        """
        :item: index of dataset to be fetched
        :return: (dataset_meta_feature vector for this dataset ,
         this dataset's tensor (n_slices, n_algo)). the second entry is essentially
         all the available fidelity slices / learning curve (first dim/index) for all
         algorithms (second dim: columns)
        """
        it = self.split[item]
        return self.meta_dataset[it], self.lc[it]  # FIXME: activate, self.meta_algo[a],

    def __len__(self):
        return len(self.split)


class Dataset_Joint_Taskswise(Dataset_Join_Dmajor):
    def __init__(self, meta_dataset: DatasetMetaFeatures, lc: Dataset_LC,
                 meta_algo: Optional[AlgorithmMetaFeatures] = None, split=None, is_test_set: bool = False):
        super(Dataset_Joint_Taskswise, self).__init__(meta_dataset, lc, meta_algo)
        self.split = np.asarray(self.split)
        lc_dims = self.lc.transformed_df.shape
        meta_algos_dims = self.meta_algo.transformed_df.shape
        meta_dataset_dims = self.meta_dataset.transformed_df.shape

        assert lc_dims[0] == meta_dataset_dims[0], f'Number of datasets from lc and meta_dataset must equal,' \
                                                   f'However, they are {lc_dims[0]} and {meta_algos_dims[0]}' \
                                                   f'respectively.'
        assert lc_dims[-1] == meta_algos_dims[0], f'Number of algorihms from lc and meta_algos_dims must equal,' \
                                                  f'However, they are {lc_dims[0]} and {meta_algos_dims[0]}' \
                                                  f'respectively.'
        self.num_datasets = len(self.split)
        self.num_algos = lc_dims[-1]

        self.is_test_set = is_test_set
        if self.is_test_set and split is None:
            raise ValueError('If the dataset is test set, it must contain the information training sets')
        if self.is_test_set and split is not None:
            self.training_sets = np.setdiff1d(np.arange(lc_dims[0]), self.split)
            if len(self.training_sets) == 0:
                raise ValueError('If the dataset is test set, it must contain the information training sets')

    """
    A Dataset
    """

    def __getitem__(self, item):
        """
        This dataset returns the following items:
        X_lc: torch.Tensor
            learning curve of the algorithm configuration on all the meta datasets with shape [N_dataset, L, N_feature]
        X_meta_features: torch.Tensor
            meta features of the meta dataset with shape [N_dataset, N_metafeatures]
        algo_features: torch.Tensor
            algorithm features with shape [N]
        y_meta_features: torch.Tensor
            meta features of the test set with shape [N_dataset, N_metafeatures]
        y_lc: torch.Tensor
            learning curves on the test datasets with shape [N_algo, L, N_features]
        """
        idx_dataset = item % self.num_datasets
        algo = item // self.num_datasets

        if not self.is_test_set:
            dataset_y = self.split[idx_dataset]
            dataset_X = self.split[torch.arange(len(self.split)) != idx_dataset]
        else:
            dataset_y = self.split[idx_dataset]
            dataset_X = self.training_sets

        tgt_meta_features = self.meta_dataset.transformed_df[dataset_y]

        X_meta_features = self.meta_dataset.transformed_df[dataset_X]
        X_lc = self.lc.transformed_df[dataset_X][:, :, [algo]]

        query_algo_idx = torch.arange(self.num_algos) != algo

        tgt_algo_features = self.meta_algo.transformed_df[algo]
        query_algo_features = self.meta_algo.transformed_df[query_algo_idx]

        y_lc = self.lc.transformed_df[dataset_y]

        tgt_algo_lc = y_lc[:, [algo]]
        query_algo_lc = y_lc[:, query_algo_idx]

        X = {'X_lc': X_lc,
             'X_meta_features': X_meta_features,
             'tgt_algo_features': tgt_algo_features,
             'query_algo_features': query_algo_features,
             'tgt_meta_features': tgt_meta_features,
             'query_algo_lc': query_algo_lc
             }

        y = {'tgt_algo_lc': tgt_algo_lc}

        return X, y

    def __len__(self):
        return self.num_datasets * self.num_algos
