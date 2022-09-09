from typing import Optional, Callable, List

import torch
from torch.utils.data import Dataset

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


def mask_lcs_randomly(lc_tensor, dataset=None):
    """# FIXME: move me to utils.masking.py
    Given a dataset's learning curves determine a random index for each
    curve, and set all values after that index to 0.
    :param lc_tensor: [n_fidelities, n_algos]
    :param dataset: if necessary, this allows access to the dataset object.
    :return: [n_fidelities, n_algos]
    """
    n_algos, n_fidelities = lc_tensor.shape
    mask = torch.zeros_like(lc_tensor)
    mask_idx = torch.randint(0, n_fidelities, (n_algos,)).view(-1, 1)
    for i, idx in enumerate(mask_idx):
        mask[i, 0:idx] = 1
    return lc_tensor * mask, mask.bool()


class Dataset_Join_Dmajor(Dataset):
    def __init__(
            self,
            meta_dataset: DatasetMetaFeatures,
            lc: Dataset_LC,
            meta_algo: Optional[AlgorithmMetaFeatures] = None,
            split: List[int] = None,
            masking_fn: Optional[Callable] = None,

    ):
        """
        Dataset, joining Dataset Meta features, Algorithm Meta features and the
        Learning Curves. The resulting iterator presents instances by dataset major; i.e.
        presenting the getitem index refers to the dataset to be fetched.
        Args:
            meta_dataset: meta features with size [n_datasets, n_features]
            lc: lc bench dataset with size [n_dataset, n_features, n_fidelites]
            meta_algo:
            split:
        """
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo
        self.lc = lc
        self.masking_fn = masking_fn

        if split is not None:
            self.split = split
        else:
            self.split = list(range(len(self.meta_dataset)))

    def __getitem__(self, item):
        """
        :item: int. Index of dataset to be fetched
        :return: tuple[dict[str,torch.Tensor], dict[str,torch.Tensor]]: X, y,
        where X is a dict of dataset meta features and the (randomly masked) learning curves,
        and y is a dict of the final fidelity of the learning curves.
        """
        it = self.split[item]

        # if masking strategy is supplied:
        if self.masking_fn is not None:
            lc_tensor, lc_values_observed = self.masking_fn(self.lc[it])

        else:
            lc_tensor = self.lc[it]
            lc_values_observed = torch.ones_like(lc_tensor, dtype=torch.bool)

        X = {
            "dataset_meta_features": self.meta_dataset[it],
            "learning_curves": lc_tensor,
            "lc_values_observed": lc_values_observed,
            # "hp":self.meta_algo[a], # fixme: not needed, since constant during training in
            #  columnwise
        }

        y = {"final_fidelity": self.lc[it, :, -1]}

        return X, y

    def __len__(self):
        return len(self.split)
