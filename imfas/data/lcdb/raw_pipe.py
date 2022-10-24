from itertools import product
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from lcdb import get_all_curves, get_meta_features


class LCDB_API:

    def __init__(self, root: Path):
        self.path_preprocessed = root / 'preprocessed' / 'lcdb'
        self.path_preprocessed.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        if self.df is not None:
            # find the set of learners per dataset in df
            self.df.reset_index(inplace=True)
            datasets, algos = set(self.df.openmlid), set(self.df.learner)
            self.df.set_index(['openmlid', 'learner'], inplace=True)
            avail_combinations = set(self.df.index)
            all_combinations = set(product(datasets, algos))
            missing = all_combinations - avail_combinations

            str_missing = f"Missing {len(missing) / len(all_combinations)} % dataset-algorithm " \
                          f"combinations,\n" \
                          f"Missing the combinations {missing}\n\n"

            str_message = f'Available datasets: {len(self.openmlids)}\n' \
                          f'Available learners: {len(self.learners)}\n' \
                          f'Available fidelity levels: {len(self.fidelity)}\n' \
                          f'Fidelity levels:\n {self.fidelity.values}\n'

        return f"LCDB_API(root={self.path_preprocessed})\n" + str_message + str_missing

    def load_lcs(self, metric='accuracy'):
        self.df = get_all_curves(metric=metric)
        self.df.sort_index(inplace=True)

    def load_meta_features(self):
        """Load the dataset meta features conditioned on the available datasets."""
        assert self.openmlids is not None, "Please run preprocess_to_raw first."

        df_meta = get_meta_features()

        # add the missing values to the meta feature df
        datasets = []
        for openmlid in set(self.df.index.levels[0].values) - set(df_meta.openmlid.values):
            datasets.append(openml.datasets.get_dataset(int(openmlid), download_data=False))

        qualities = {m.id: m.qualities for m in datasets}
        dataset_meta_features = pd.DataFrame.from_dict(qualities).T
        dataset_meta_features['openmlid'] = dataset_meta_features.index
        dataset_meta_features = dataset_meta_features[list(
            set(dataset_meta_features.columns).intersection(set(df_meta.columns)))]

        self.df_meta = df_meta[df_meta.openmlid.isin(self.openmlids)]
        self.df_meta = pd.concat([self.df_meta, dataset_meta_features], ignore_index=True)
        self.df_meta.set_index('openmlid', inplace=True)
        self.df_meta.sort_index(inplace=True)

    def save(self):
        """Save the preprocessed data to disk."""

        self.tables = dict()

        assert all(self.df.index.levels[0].values == self.df_meta.index.values), \
            'Openmlids do not match for df and df_meta.'
        for col in ['train', 'valid', 'test']:
            col_name = f'score_{col}'

            self.tables[col_name] = pd.pivot(self.df, columns='size_train', values=col_name)

            self.tables[col_name].to_hdf(
                self.path_preprocessed / f"{col}_lcs.h5",
                key="dataset",
                mode="w"
            )

        # self.df.to_csv(self.path_preprocessed / f'{name}.csv')
        self.df_meta.to_csv(self.path_preprocessed / f'lcdb_meta.csv')

    def find_subset_tensor(self, threshold=34, inner_seed=0, outer_seed=0, size_test=5000):
        """
        Find the relevant subset of data, that forms a Tensor of learning curves.

        :param threshold: upper bound on the power of sqrt2 for the fidelity (training-set-size).
        :param inner_seed: Seed for the inner split of the data.
        :param outer_seed: Seed for the outer split of the data.
        :param size_test: Size of the test set.
        """

        # these two algorithms are removed, as they drastically reduce the
        # number of available datasets.
        removals = {'sklearn.naive_bayes.MultinomialNB',
                    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'}
        self.learners = set(self.df.learner) - removals

        sqrt2 = np.sqrt(2)
        self.fidelity = pd.Series([int(sqrt2 ** k) for k in np.arange(start=8, stop=threshold, )])

        # subset of df with frequent learners & fidelity of sqrt**k only!
        df = self.df[self.df.learner.isin(self.learners)][self.df.size_train.isin(self.fidelity)]
        df = df[(df.inner_seed == inner_seed) &
                (df.outer_seed == outer_seed) &
                (df.size_test == size_test)]

        # find the set of learners per dataset that have been observed sufficiently often
        df1 = df.groupby(['openmlid', 'size_train', ]).count()

        # fixme: this does not consider the varying test size yet!
        # fixing the test size to the most frequent (i.e. 5000) results in 16 Datasets.
        # df.size_test.value_counts()

        # finding those datasets that have been observed at all fidelity levels
        algo_threshold = len(self.learners)
        a = df1.groupby(['openmlid']).agg(np.mean).learner >= algo_threshold
        self.openmlids = a.index[a].values

        self.df = df[df.openmlid.isin(self.openmlids)]
        self.df.set_index(['openmlid', 'learner'], inplace=True)

        # actually observed fidelity levels:
        self.fidelity = pd.Series(sorted(set(df.size_train)))


from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


def raw_pipe(*args, **kwargs):
    # make it a hydra pipe again:

    kwargs.pop('enable')
    cfg = DictConfig(kwargs)

    lcdb_pipe = LCDB_API(root=Path(__file__).parents[3] / 'data')

    metric = cfg.pop('metric')

    lcdb_pipe.load_lcs(metric=metric)
    lcdb_pipe.find_subset_tensor(**cfg)
    lcdb_pipe.load_meta_features()

    logger.info(lcdb_pipe)

    lcdb_pipe.save()


if __name__ == '__main__':
    # TODO hydra_wrapper & config
    lcdb_pipe = LCDB_API(root=Path(__file__).parents[3] / 'data')
    lcdb_pipe.load_lcs(metric='accuracy')
    lcdb_pipe.find_subset_tensor(threshold=34, inner_seed=0, outer_seed=0, size_test=5000)
    lcdb_pipe.load_meta_features()
    print(lcdb_pipe)
    lcdb_pipe.save()
