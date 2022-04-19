import pandas as pd
from torch.utils.data import Dataset

from mf_gravitas.data.preprocessings.lc_slice import LC_TimeSlice
import pdb

class Dataset_LC(Dataset):
    def __init__(self, path, metric, transforms):
        self.df = pd.read_hdf(path, mode='r')
        self.df = self.df.xs(key=metric)

        # consider: is this possible by read in? - to reduce memory overhead

        self.multidex = self.df.index
        self.transforms = transforms

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transforms.transform(self.df).T
            else:
                self.transformed_df = self.transforms.fit(self.df).T
            self.df = self.df[51].unstack().T  # transform the df appropriately
            # self.transformed_df = self.transformed_df

            # for getitem;
            last_time_slice = [trans for trans in self.transforms
                               if isinstance(trans, LC_TimeSlice)][-1]

            if last_time_slice:
                # time slices create a new dataframe, whose indicies must be known
                # to get some sensible indexing going
                self.columns = last_time_slice.columns
                self.index = last_time_slice.index

    def __getitem__(self, item: int):
        # fixme: this won't be applicable no more when transform has ToTensor
        # return single learning curve:
        # self.df.loc[item, :]  # item:tuple e.g. ('APSFailure', '0')

        return self.transformed_df[item]

    def __len__(self):
        return len(self.df.nrows)


if __name__ == '__main__':
    import os

    current = os.getcwd().split('/')
    file = '~/PycharmProjects/AlgoSelectionMF/data/preprocessed/LCBench/logs.h5'
    lc = Dataset_LC(file, 'Train/val_accuracy', None)

    # lc.df.loc[('APSFailure', '0'), :]

    # select a time slice
    # lc.df[51].unstack().T

    from mf_gravitas.data.preprocessings.table_transforms import ToTensor
    from mf_gravitas.data.preprocessings.transformpipeline import TransformPipeline

    pipe = TransformPipeline([LC_TimeSlice(slice=51), ToTensor()])
    lc = Dataset_LC(file, 'Train/val_accuracy', pipe)

    # lc[('APSFailure', '0')]

    lc[0]
