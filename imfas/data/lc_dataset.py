import pandas as pd
from torch.utils.data import Dataset

from imfas.data.preprocessings.lc_slice import LC_TimeSlices
from imfas.data.preprocessings.transformpipeline import TransformPipeline


class Dataset_LC(Dataset):
    def __init__(self, path, transforms: TransformPipeline, metric: str = "None"):
        """
        :param metric: Lcbench needs another specifier to subset the dataset.
        """
        self.df = pd.read_hdf(path, mode="r")

        if metric != "None":
            self.df = self.df.xs(key=metric)

        # consider: is this possible by read in? - to reduce memory overhead

        self.multidex = self.df.index
        self.transforms = transforms

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transforms.transform(self.df)
            else:
                self.transforms = self.transforms.fit(self.df)
                self.transformed_df = self.transforms.transform(self.df)

    def __getitem__(self, item: int):
        """
        :param item: int. index of dataset to be queried
        :returns: tensor of shape (n_fidelities, n_algorithms)
        """
        # FIXME: this won't be applicable no more when transform has ToTensor
        # return single learning curve:
        # self.df.loc[item, :]  # item:tuple e.g. ('APSFailure', '0')

        return self.transformed_df[item]

    def __len__(self):
        # Dataset major; i.e. dataset dimension!
        return self.shape[0]

    @property
    def shape(self):
        return self.transformed_df.shape


if __name__ == "__main__":
    import os

    current = os.getcwd().split("/")
    file = "~/PycharmProjects/AlgoSelectionMF/data/preprocessed/LCBench/logs.h5"
    lc = Dataset_LC(file, "Train/val_accuracy", None)

    # lc.df.loc[('APSFailure', '0'), :]

    # select a time slice
    # lc.df[51].unstack().T

    from imfas.data.preprocessings.table_transforms import ToTensor
    from imfas.data.preprocessings.transformpipeline import TransformPipeline

    pipe = TransformPipeline([LC_TimeSlices(slice=51), ToTensor()])
    lc = Dataset_LC(file, "Train/val_accuracy", pipe)

    # lc[('APSFailure', '0')]

    lc[0]
