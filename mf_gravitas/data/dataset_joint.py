from copy import deepcopy
from random import randint

import pandas as pd
from torch.utils.data import Dataset


class Dataset_Join(Dataset):
    def __init__(self, meta_dataset, meta_algo, lc, competitors: int = 0):
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo
        self.lc = lc

        self.competitors = competitors

        # Fixme: add consistency checks on rows & columns!

        # fixme: this is brittle and depends on the lc.transformed_df format after the pipe!
        # it also assumes that meta_dataset & meta_algo have the exact same ordering
        self.dataset_names, self.algo_names = self.lc.columns, self.lc.index
        self.multidex = deepcopy(lc.multidex)
        self.multidex = self.multidex.set_levels([
            list(range(len(self.dataset_names))),
            list(range(len(self.algo_names)))
        ])

        # be aware of row columns in:
        # self.lc.df[51].unstack().T

    def __getitem__(self, item):
        """
        sync getitem across the multiple dataset classes.
        """
        D_m, A_m, a_p = self.__get_single__(item)
        if self.competitors > 0:
            competitors = self.__get_competitors__(item)

            return (D_m, a_p), competitors  # fixme: algo meta features are diabled
        else:
            return (D_m, a_p), (None, None)

    def __get_single__(self, item):
        # parse the index & allow it to fetch multiple vectors at once
        d, a = self.multidex[item]
        # fixme: indexing depends on the transformations applied
        #  in particularly troubeling is lc, ince it is a time slice!
        return self.meta_dataset[d], self.meta_algo[a], self.lc[a]

    def __get_multiple__(self, items):
        """
        Fetch multiple items at once (output is like single, but with
        stacked tensors)
        """
        d, a = zip(*self.multidex[items])
        d, a = list(d), list(a)
        # fixme: indexing depends on the transformations applied
        #  in particularly troubeling is lc, ince it is a time slice!
        return self.meta_dataset[d], self.meta_algo[a], self.lc[a]

    def __get_competitors__(self, item):
        # Consider: Creating the competitor set might be the bottleneck
        competitors = [randint(0, self.__len__()) for c in range(self.competitors)]
        competitors = [c for c in competitors if c != item]

        # ensure, we will never hit the same item as competitor
        if not competitors:
            while competitors:
                val = randint(0, self.__len__())
                if val != item:
                    competitors = [randint(0, self.__len__())]

        return self.__get_multiple__(competitors)

    def __len__(self):
        return len(self.multidex)


class Dataset_Join_Split(Dataset_Join):
    def __init__(self, splitindex: list[int], *args, **kwargs):
        """
        Convenience wrapper around Dataset_Join to
        Deterministically split it into train and test sets based on splitindex.
        :param splitindex: index of the datasets that are to be kept
        """
        super(Dataset_Join_Split, self).__init__(*args, **kwargs)
        self.splitindex = splitindex

        self.multidex = pd.MultiIndex.from_tuples(
            [(d, a) for d, a in self.multidex if d in splitindex],
            names=['dataset', 'algorithm'])
