import pandas as pd
from torch.utils.data import Dataset


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
    def __init__(self, file, metirc):
        df = pd.read_hdf(file, mode='r')

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


if __name__ == '__main__':
    import os

    current = os.getcwd().split('/')

    file = '/'.join(
        current[:-3] + ['data', 'preprocessed', 'data_2k', 'train_cross_entropy_curves.json'])
    Dataset_LC(file)
