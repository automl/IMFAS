from typing import Callable, List, Optional

import torch
from torch.utils.data import Dataset

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


class Dataset_Join_Dmajor(Dataset):
    def __init__(
            self,
            learning_curves: Dataset_LC,
            meta_dataset: DatasetMetaFeatures,
            meta_algo: Optional[AlgorithmMetaFeatures] = None,
            split: List[int] = None,
            masking_fn: Optional[Callable] = None,
            **kwargs
    ):
        """
        Dataset, joining Dataset Meta features, Algorithm Meta features and the
        Learning Curves. The resulting iterator presents instances by dataset major; i.e.
        presenting the getitem index refers to the dataset to be fetched.
        Args:
            meta_dataset: meta features with size [n_datasets, n_features]
            learning_curves: dataset_LC with size [n_dataset, n_features, n_fidelites]
            meta_algo:
            split: list of indicies, that this dataset will have access to (i.e. which
            datasets are available)
        """

        self.learning_curves = learning_curves
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo
        self.masking_fn = masking_fn
        self.kwargs = kwargs

        assert all([v.shape == self.learning_curves.shape for v in kwargs.values()])
        assert self.learning_curves.shape[0] == meta_dataset.shape[0]
        if meta_algo is not None:
            assert self.learning_curves.shape[1] == meta_algo.shape[1]

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
            lc_tensor, mask = self.masking_fn(self.learning_curves[it])


        else:
            lc_tensor = self.learning_curves[it]
            mask = torch.ones_like(lc_tensor, dtype=torch.bool)

        X = {
            "dataset_meta_features": self.meta_dataset[it],
            "learning_curves": lc_tensor,
            "mask": mask,
            **self.kwargs

            # "hp":self.meta_algo[a], # fixme: not needed, since constant during training in

        }

        y = {"final_fidelity": self.learning_curves[it, :, -1]}

        return X, y

    def __len__(self):
        return len(self.split)

    def __repr__(self):
        message = f"Dataset_Join_Dmajor(split: {self.split})\n" \
                  f"Shapes: " \
                  f"\n\tDatasetMeta: {self.meta_dataset.shape} " \
                  f"\n\tDatasetLC: {self.lcs.shape}"

        if self.meta_algo is not None:
            message += f"\n\tAlgorithmMeta: {self.meta_algo.shape}"

        return message


class Dataset_Join_Test(Dataset_Join_Dmajor):
    def __init__(self, test_curves, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_curves = test_curves

    def __getitem__(self, item):
        """
        :item: int. Index of dataset to be fetched
        :return: tuple[dict[str,torch.Tensor], dict[str,torch.Tensor]]: X, y,
        Notably, y is enriched with the final test_slice of the learning curve,
        to check the generalization performance of the model.
        """
        it = self.split[item]
        X, y = super().__getitem__(item)
        y = {"final_fidelity": self.test_curves[it, :, -1]}
        return X, y


if __name__ == "__main__":
    from pathlib import Path
    import imfas.data.preprocessings as prep

    root = Path(__file__).parents[3]

    dataset_name = "LCBench"
    data_path = root / 'data' / 'raw' / dataset_name

    pipe_lc = prep.TransformPipeline(
        [prep.Column_Mean(), prep.Convert(), prep.LC_TimeSlices(slices=[0, 1, 2, 3])]
    )

    pipe_meta = prep.TransformPipeline(
        [prep.Zero_fill(), prep.Convert(), prep.ToTensor(), prep.ScaleStd()]
    )

    pipe_algo = prep.TransformPipeline(
        [prep.Zero_fill(),
         prep.Drop(
             ['imputation_strategy', 'learning_rate_scheduler', 'loss', 'network',
              'normalization_strategy', 'optimizer', 'activation', 'mlp_shape', ]),
         prep.Replace(columns=['num_layers'], replacedict={'True': 1}),
         prep.Convert(),
         prep.ToTensor(),
         prep.ScaleStd()]
    )

    D = Dataset_Join_Dmajor(
        meta_dataset=DatasetMetaFeatures(
            path=data_path / 'meta_features.csv',
            transforms=pipe_meta),
        lc=Dataset_LC(
            path=data_path / 'logs_subset.h5',
            transforms=pipe_lc),
        meta_algo=AlgorithmMetaFeatures(
            path=data_path / 'config_subset.csv',
            transforms=pipe_algo),
        split=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    D[0]
