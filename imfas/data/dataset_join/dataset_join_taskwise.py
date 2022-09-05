import torch

from imfas.data.dataset_join.dataset_join_taskwiseranking import Dataset_Joint_TaskswiseRanking


class Dataset_Joint_Taskswise(Dataset_Joint_TaskswiseRanking):
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
        if self.is_test_set:
            return super(Dataset_Joint_Taskswise, self).__getitem__(item)
        idx_dataset = item % self.num_datasets
        algo = item // self.num_datasets

        dataset_y = self.split[idx_dataset]
        dataset_X = self.split[torch.arange(len(self.split)) != idx_dataset]

        tgt_meta_features = self.meta_dataset.transformed_df[dataset_y]

        X_meta_features = self.meta_dataset.transformed_df[dataset_X]
        X_lc = self.lc.transformed_df[dataset_X][:, :, [algo]]

        # all other algorithms next to the target algorithm
        query_algo_idx = torch.arange(self.num_algos) != algo

        tgt_algo_features = self.meta_algo.transformed_df[algo]
        query_algo_features = self.meta_algo.transformed_df[query_algo_idx]

        y_lc = self.lc.transformed_df[dataset_y]

        tgt_algo_lc = y_lc[:, [algo]]
        query_algo_lc = y_lc[:, query_algo_idx]

        X = {
            'X_lc': X_lc,
            'X_meta_features': X_meta_features,
            'tgt_algo_features': tgt_algo_features,
            'query_algo_features': query_algo_features,
            'tgt_meta_features': tgt_meta_features,
            'query_algo_lc': query_algo_lc
        }

        y = {'tgt_algo_lc': tgt_algo_lc}

        return X, y

    def __len__(self):
        return self.num_datasets if self.is_test_set else self.num_datasets * self.num_algos


from typing import Optional
from torch.utils.data import Dataset

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


class Dataset_Joint_Taskwise2(Dataset):
    #
    def __init__(
            self,
            meta_dataset: DatasetMetaFeatures,
            lc: Dataset_LC,
            meta_algo: Optional[AlgorithmMetaFeatures] = None,
            split=None
    ):
        """
        A Dataset, assuming there is a cuboid, where the vertical dim are the algorithms,
        the horizontal dim are the datasets and the depth dim is their learning curve observed
        at different fidelities.

        :split: list of dataset indicies, that are accessible to this Dataset class. This is
        used to create a holdout /test set
        """
        self.meta_dataset = meta_dataset
        self.lc = lc
        self.meta_algo = meta_algo

        if split is None:
            split = set(i for i in range(self.num_datasets))

        # descriptives
        self.n_datasets = len(split)
        self.n_algos = self.meta_algo.transformed_df.shape[0]
        self.n_algo_features = self.meta_algo.transformed_df.shape[1]
        self.n_meta_features = self.meta_dataset.transformed_df.shape[1]
        self.len_lc = self.lc.transformed_df.shape[2]

        # identifier indicies for the dataset-algorithm tuples
        self.datasets = set(split)
        self.algos = set(range(self.n_algos))
        self.index = [(d, a) for d in self.datasets for a in self.algos]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item: int):
        """
        :param item: idx. Notice, that this idx is translated into a dataset algorithm tuple (D_i, A_j)

        :return: X, y
        """

        d, a = self.index[item]

        # current algorithm, dataset combination
        d_meta = self.meta_dataset.transformed_df[d]
        y_lc = self.lc.transformed_df[d][:, [a]]  # fixme: is this the target algorithm's lc?
        # fixme: should we present the "label" i.e. unmasked lc and the same but masked
        #  lc as "training datapoint"

        # Meta context
        hp, otheralgos_curves = self.get_other_algorithms_opinions(d, a)
        meta_features, lcs = self.get_target_algorithms_known_lcs(d, a)

        X = {}
        y = {}

        return X, y

    def get_other_algorithms_opinions(self, d, inquisitor):
        """
        Select a subset of algorihtms' learning curves that are not the target algorithm and their
        meta features. (This is the
        :param d: dataset index
        :param inquisitor: algorithm index, that is the target algorithm, who is asking for
        opinions.

        """
        # TODO random subsample?
        As = list(self.algos / inquisitor)  # consider: do we need the list conversion?
        lcs = self.lc.transformed_df[As, d]  # fixme: check index selection works!

        observed_curves = self.mask_lc(lcs)

        hp = self.meta_algo.transformed_df[As]
        return hp, observed_curves

    def get_target_algorithms_known_lcs(self, active_dataset, inquisitor):
        """
        Select a subset of meta-known learning curves from the target algorihm,
        contextualized by the dataset meta features it acted on.
        :param active_dataset: the dataset index on which the target algorithm's performance
        is to be predicted.
        :param inquisitor: the algorithm index, that is the target algorithm of
        which we want to know its past rollouts.
        """
        # TODO random subsample?
        ds = list(self.datasets / active_dataset)
        lcs = self.lc.transformed_df[inquisitor, ds]
        meta_features = self.meta_dataset.transformed_df[ds]

        return meta_features, lcs

    def mask_lc(self, lc):
        # TODO CONSIDER moving this out of the class to allow for complex schedules!
        # TODO move difan's trainer masking code here

        # select a random fidelity
        pass
