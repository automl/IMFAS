from random import randint

from torch.utils.data import Dataset

from imfas.data.algorithm_meta_features import AlgorithmMetaFeatures
from imfas.data.dataset_meta_features import DatasetMetaFeatures
from imfas.data.lc_dataset import Dataset_LC


# FIXME: Refactor joint - gravity, multiindex & compettitor sampling
class Dataset_Join_Gravity(Dataset):
    def __init__(
        self, meta_dataset: DatasetMetaFeatures, meta_algo: AlgorithmMetaFeatures, lc: Dataset_LC, competitors: int = 0
    ):
        self.meta_dataset = meta_dataset
        self.meta_algo = meta_algo
        self.lc = lc

        self.competitors = competitors

        # FIXME: add consistency checks on rows & columns!

        # FIXME: this is brittle and depends on the lc.transformed_df format after the pipe!
        # it also assumes that meta_dataset & meta_algo have the exact same ordering
        self.dataset_names, self.algo_names = self.lc.columns, self.lc.index

        # LC Style
        # self.multidex = deepcopy(lc.multidex)
        # self.multidex = self.multidex.set_levels([
        #     list(range(len(self.dataset_names))),
        #     list(range(len(self.algo_names)))
        # ])
        #

        # LC Slice Style
        # This index is useful with
        self.multidex = list(
            (d, a)
            for d in range(len(self.meta_dataset.transformed_df))
            for a in range(len(self.meta_algo.transformed_df))
        )

        # be aware of row columns in:
        # self.lc.df[51].unstack().T

    def __getitem__(self, item):
        """
        sync getitem across the multiple dataset classes.
        """
        D_m, A_m, a_p = self.__get_single__(item)
        if self.competitors > 0:
            competitors = self.__get_competitors__(item)

            return (D_m, a_p), competitors  # FIXME: algo meta features are disabled
        else:
            return (D_m, a_p), (None, None)

    def __get_single__(self, item):
        d, a = self.multidex[item]
        # FIXME: indexing depends on the transformations applied
        #  in particularly troubling is lc, since it is a time slice!
        return self.meta_dataset[d], self.meta_algo[a], self.lc[a]

    def __get_multiple__(self, items):
        """
        Fetch multiple items at once (output is like single, but with
        stacked tensors)
        """
        # parse the index & allow it to fetch multiple vectors at once
        # LC Slice style
        l = [self.multidex[i - 1] for i in items]
        d, a = zip(*l)

        # Consider using this when moving from LC Slice to LC
        # LC Style
        # d, a = zip(*self.multidex[items])

        d, a = list(d), list(a)
        # FIXME: indexing depends on the transformations applied
        #  in particularly troubling is lc, since it is a time slice!
        return self.meta_dataset[d], self.lc[d]  # self.meta_algo[a], # FIXME add in algo meta

    def __get_competitors__(self, item):
        # Consider: Creating the competitor set might be the bottleneck
        competitors = [randint(0, self.__len__()) for c in range(self.competitors)]
        competitors = [c for c in competitors if c != item]

        # ensure, we will never hit the same item as competitor
        if len(competitors) != self.competitors:
            while len(competitors) != self.competitors:
                val = randint(0, self.__len__())
                if val != item:
                    competitors.append(randint(0, self.__len__()))

        return self.__get_multiple__(competitors)

    def __len__(self):
        return len(self.meta_dataset.transformed_df) * len(self.meta_algo.transformed_df)


class Dataset_Join_Split(Dataset_Join_Gravity):
    def __init__(self, splitindex: list[int], *args, **kwargs):
        """
        Convenience wrapper around Dataset_Join_Gravity to
        Deterministically split it into train and test sets based on splitindex.
        :param splitindex: index of the datasets that are to be kept
        """
        super(Dataset_Join_Split, self).__init__(*args, **kwargs)
        self.splitindex = splitindex

        # Consider using this, when switching LC slice or LC!
        # self.multidex = pd.MultiIndex.from_tuples(
        #     [(d, a) for d, a in self.multidex if d in splitindex],
        #     names=['dataset', 'algorithm'])

        # This index is useful with
        self.multidex = list(
            (d, a) for d in self.splitindex for a in range(len(self.meta_dataset))
        )  # range(len(self.meta_algo.transformed_df)))

    def __len__(self):
        return len(self.splitindex) * len(self.meta_dataset)
