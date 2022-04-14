import pandas as pd
from torch.utils.data import Dataset

from mf_gravitas.data.preprocessings.lc_slice import LC_TimeSlice


# Consider:
#  1. separate classes dataset-sources by
#     (dataset_meta, algo_meta, Learning_curve_fidelity, final performance)
#     since each of these might need different preprocessings.
#  1.1 shall the learning curve be iteratively presented or as a slice until the current
#      reveal status? - probably rather the former then the latter to support
#      the budget allocation and reveal process, which is more realistic (decide when to evaluate)
#      there is however the issue of having recordered already, rather than
#      letting the agent decide continuous budgets on his own.
#  1.2 in case of splitting the dataset into its components, then how & where to
#      "sync" the dataset query - in particular, when shuffling the dataset?
#  2. How & where to configure and apply the respective preprocessings?
#  3. Be careful with train test split! we need to be able mask the loaded json
#     and ideally not hold the dataset multiple times in memory!
#  4. how to get fidelity information in there (if. e.g. unevenly spaced subset sizes)


class Dataset_LC(Dataset):
    def __init__(self, file, metric, transforms):
        self.df = pd.read_hdf(file, mode='r')
        self.df = self.df.xs(key=metric)

        self.transforms = transforms

        if transforms is not None:
            if transforms.fitted:
                self.transformed_df = self.transforms.transform(self.df)
            else:
                self.transformed_df = self.transforms.fit(self.df)

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
