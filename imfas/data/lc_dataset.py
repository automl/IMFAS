import pandas as pd
from torch.utils.data import Dataset

from imfas.data.preprocessings.transformpipeline import TransformPipeline
from imfas.data.preprocessings.lc_slice import LC_TimeSlices


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
                # FIXME: .T is depreciated for more than two dimensions
                self.transformed_df = self.transforms.transform(self.df).T
            else:
                # FIXME: .T is depreciated for more than two dimensions
                self.transforms = self.transforms.fit(self.df)
                self.transformed_df = self.transforms.transform(self.df).T
            self.df = self.df[self.df.columns[-1]].unstack().T  # transform the df appropriately
            # self.transformed_df = self.transformed_df

            # for getitem;
            last_time_slice = [trans for trans in self.transforms if isinstance(trans, LC_TimeSlices)][-1]

            if last_time_slice:
                # time slices create a new dataframe, whose indicies must be known
                # to get some sensible indexing going
                self.columns = last_time_slice.columns
                self.index = last_time_slice.index


    def __getitem__(self, item: int):
        """
        :param item: index of dataset to be queried
        :returns: tensor of shape (n_fidelities, n_algorithms)
        """
        # FIXME: this won't be applicable no more when transform has ToTensor
        # return single learning curve:
        # self.df.loc[item, :]  # item:tuple e.g. ('APSFailure', '0')

        return self.transformed_df[item]

    def __len__(self):
        return len(self.df)


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
