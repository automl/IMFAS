from imfas.data.dataset_join.dataset_join_Dmajor import Dataset_Join_Dmajor


# TODO rewrite this as a single batch (of size self.split) with the fidelity slice as yaml config
#  dataset.slices = [52] for instance. with dataset.dataloader.batch_size = len(self.split) -->
#  which would need to be dynamically computed though.
class Dataset_join_classicAS(Dataset_Join_Dmajor):
    def __init__(self, slice: int = -1, *args, **kwargs):
        """
        Assuming that the model is static wrt to the fidelity, the dataset should
        return all the training/testing data at once.

        :param slice: int. The slice of the learning curve to be used as performance target.
        """
        super(Dataset_join_classicAS, self).__init__(*args, **kwargs)

        self.slice = slice

    def __getitem__(self, item):
        """
        :item: int. Index of dataset to be fetched
        :return: tuple[dict[str,torch.Tensor], dict[str,torch.Tensor]]: X, y,
        where X is a dict of dataset meta features and the (randomly masked) learning curves,
        and y is a dict of the final fidelity of the learning curves.
        """
        X = {"dataset_meta_features": self.meta_dataset[self.split],
             "fidelity": self.lc[self.split, :, self.slice]}

        y = {"final_fidelity": self.lc[self.split, :, self.slice], }

        return X, y

    def __len__(self):
        return 1


if __name__ == "__main__":
    from pathlib import Path
    import imfas.data.preprocessings as prep
    from imfas.data.dataset_meta_features import DatasetMetaFeatures
    from imfas.data import Dataset_LC

    root = Path(__file__).parents[3]

    dataset_name = "LCBench"
    data_path = root / 'data' / 'raw' / dataset_name

    pipe_lc = prep.TransformPipeline(
        [prep.Column_Mean(), prep.Convert(), prep.LC_TimeSlices(slices=[3])]
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

    D = Dataset_join_classicAS(
        meta_dataset=DatasetMetaFeatures(
            path=data_path / 'meta_features.csv',
            transforms=pipe_meta),
        lc=Dataset_LC(
            path=data_path / 'logs_subset.h5',
            transforms=pipe_lc,
            metric='Train/train_accuracy'),

        split=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    D[0]
