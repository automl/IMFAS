from typing import Optional

from torch.utils.data import Dataset

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


class Dataset_Join_Dmajor(Dataset):
    def __init__(
            self,
            meta_dataset: DatasetMetaFeatures,
            lc: Dataset_LC,
            meta_algo: Optional[AlgorithmMetaFeatures] = None,
            split=None
    ):
        """
        Dataset, joining Dataset Meta features, Algorithm Meta features and the
        Learning Curves. The resulting iterator presents instances by dataset major; i.e.
        presenting the getitem index refers to the dataset to be fetched.
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
        return {
            "dataset_meta_features": self.meta_dataset[it],
            "learning_curve": self.lc[it]
            # FIXME: activate: " "self.meta_algo[a],
        }

    def __len__(self):
        return len(self.split)
