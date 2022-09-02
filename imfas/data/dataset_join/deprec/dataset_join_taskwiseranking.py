from typing import Optional

import numpy as np
import torch

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_join.dataset_join_Dmajor import Dataset_Join_Dmajor
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


class Dataset_Joint_TaskswiseRanking(Dataset_Join_Dmajor):
    # FIXME: @ DIFAN from what i gather you could incorporate basically all of this
    #  into the Dataset_Join_Dmajor class, only changing the return of getitem to
    #  be a dict (which I admit is preferable to a tuple) and adding the algo meta
    #  features to the X variable. We could add final_fidelity = tgt_algo_lc[:, -1]
    #  to the y variable. I think this would be a good idea. We only need to be consistent on
    #  the algorithm ordering.
    """
    A Dataset for computing the ranking losses. This dataset is quite similar to the vanilla Taskswise Dataset. however,
    each of its item contain the learning curves of all the algorithms
    """

    def __init__(
            self,
            meta_dataset: DatasetMetaFeatures,
            lc: Dataset_LC,
            meta_algo: Optional[AlgorithmMetaFeatures] = None,
            split=None,
            is_test_set: bool = False
    ):

        super(Dataset_Joint_TaskswiseRanking, self).__init__(
            meta_dataset, lc, meta_algo, split=split
        )

        self.is_test_set = is_test_set  # FIXME: why do we need this?
        self.split = np.asarray(self.split)
        lc_dims = self.lc.transformed_df.shape
        meta_algos_dims = self.meta_algo.transformed_df.shape
        meta_dataset_dims = self.meta_dataset.transformed_df.shape

        # FIXME: move to parent class because this always has to be the case!
        assert lc_dims[0] == meta_dataset_dims[
            0], f'Number of datasets from lc and meta_dataset must equal,' \
                f'However, they are {lc_dims[0]} and {meta_algos_dims[0]}' \
                f'respectively.'
        assert lc_dims[-1] == meta_algos_dims[
            0], f'Number of algorihms from lc and meta_algos_dims must equal,' \
                f'However, they are {lc_dims[0]} and {meta_algos_dims[0]}' \
                f'respectively.'

        self.n_datasets = len(self.split)
        self.n_algos = lc_dims[-1]

        # FIXME: @DIFAN: remove this. Although you are right in that the data is
        #  held in both datasets instances (train & test), this is a much cleaner and obvious
        #  interface.
        if self.is_test_set and split is None:
            raise ValueError(
                'If the dataset is test set, it must contain the information of trainer sets')
        if self.is_test_set and split is not None:
            self.training_sets = np.setdiff1d(np.arange(lc_dims[0]), self.split)
            if len(self.training_sets) == 0:
                raise ValueError(
                    'If the dataset is test set, it must contain the information of trainer sets')

    def __getitem__(self, item):
        """
        This dataset returns the following items:
        X_lc: torch.Tensor
            learning curve of all the algorithm configuration on all the meta datasets with shape
            [N_dataset, L, N_algos]
        X_meta_features: torch.Tensor
            meta features of the meta dataset with shape [N_dataset, N_metafeatures]
        algo_features: torch.Tensor
            algorithm features with shape [N_algo, N_algofeatures]
        y_meta_features: torch.Tensor
            meta features of the test set with shape [N_dataset, N_metafeatures]
        y_lc: torch.Tensor
            learning curves on the test datasets with shape [N_algo, L, N_features]
        """
        # FIXME: @Difan preferred format: --------------------------------------
        #  make sure to document what you consider to be an item
        #  a dataset? an algorithm? a dataset-algorithm pair?

        # idx = self.split[item]
        # dataset_y = self.meta_dataset.transformed_df[idx]
        # dataset_x = self.meta_dataset.transformed_df[idx]

        if not self.is_test_set:
            dataset_y = self.split[item]
            dataset_X = self.split[torch.arange(len(self.split)) != item]
        else:
            dataset_y = self.split[item]
            dataset_X = self.training_sets

        # ----------------------------------------------------------------------

        # FIXME: what are target meta features? and why are they not part of
        #  the y variable
        tgt_meta_features = self.meta_dataset.transformed_df[dataset_y]

        X_meta_features = self.meta_dataset.transformed_df[dataset_X]
        X_lc = self.lc.transformed_df[dataset_X]

        algo_features = self.meta_algo.transformed_df
        y_lc = self.lc.transformed_df[dataset_y]

        X = {
            'X_lc': X_lc,
            'X_meta_features': X_meta_features,
            'algo_features': algo_features,
            'tgt_meta_features': tgt_meta_features,
        }

        y = {'y_lc': y_lc}

        return X, y

    def __len__(self):
        return self.n_datasets
