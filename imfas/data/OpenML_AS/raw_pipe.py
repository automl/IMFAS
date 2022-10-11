import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import read_sql_query

log = logging.getLogger(__name__)


class OpenML_ASLC_Sqlite:

    def __init__(self, root: Path):
        self.path_download = root / 'downloads' / 'openml'
        self.path_raw = root / 'raw' / 'openml'
        self.path_preprocessed = root / 'preprocessed' / 'openml'

        self.path_raw.mkdir(parents=True, exist_ok=True)
        self.path_preprocessed.mkdir(parents=True, exist_ok=True)
        self.path_download.mkdir(parents=True, exist_ok=True)

    def load_download(self, path):
        with sqlite3.connect(path) as dbcon:
            tables = list(
                read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])

            collected_tables = {
                tbl: read_sql_query(f"SELECT * from {tbl}", dbcon)
                for tbl in tables
            }

        # todo add train time as well...
        df = collected_tables['openmlcc18_learning_curve_data']
        df.set_index(['dataset_id', 'classifier', 'budget'], inplace=True)

        # switch out missing values with nan
        df.replace({'[]': '[nan, nan, nan]'}, inplace=True)

    def load(self, filename: [Optional[str]] = 'server_conference_iclr2022.db',
             from_download=False):

        if (self.path_raw / 'openml_aslc.pkl').exists() and not from_download:  # from
            # preprocessed
            log.info('Loading from %s', self.path_raw)
            self.df = pd.read_pickle(self.path_raw / 'openml_aslc.pkl')
        else:
            log.info('Loading from download')
            self.load_download(self.path_download / filename)

            # 'validation_accuracy_per_fold',
            self.preprocess_to_raw(
                cols=['test_accuracy_per_fold', 'train_time_s_per_fold'])
            log.debug('Saving to %s', self.path_raw / 'openml_aslc.pkl')
            self.df.to_pickle(self.path_raw / 'openml_aslc.pkl')

        self.df = self.df.unstack(level=['dataset_id', 'classifier']).T
        self.df.index.names = ['variable', 'dataset_id', 'classifier']
        self.df = self.df.sort_index(0)

        return self.df

    def preprocess_to_raw(self, cols):
        # separate the test_accuracy_per_fold column of lists into multiple columns
        new_cols = []
        for col in cols:
            new_col = col.replace('_per', '')
            n = [new_col + str(i) for i in range(1, 4)]
            new_cols.extend(n)
            self.df[n] = self.df[col].str.strip('[]').str.split(',').apply(pd.Series).astype(float)

        # FIXME: check if this drop actually works.
        self.df.drop(columns=cols, inplace=True)

    def collect_tables(self, cols):
        self.tables = dict()
        for col in cols:
            self.tables[col] = self.df.xs(key=col)
            self.tables[col].to_hdf(
                self.path_preprocessed / f"{col}_lcs.h5",
                key="dataset",
                mode="w"
            )

        return self.tables

    # FIXME: lcbench provides logs.h5 file. This is a hdf5 file. But it makes sense to collect
    # a single table that is passed to lc_dataset.py

    # def convert_to_tensor(self, cols: List[str]):
    #     # FIXME: move this to the dataset class? as preprocessing??
    #     """
    #     Convert to learning curve tensor.
    #     :cols: list of column names to convert into a single tensor. (i.e. across folds)
    #     :returns: torch.Tensor [n_datasets, n_algos, n_folds, n_budgets]
    #     """

    #
    #     # {k, v.tables
    #     return torch.stack([torch.tensor(table.values) for table in tables.values()])


if __name__ == '__main__':
    path = Path(__file__).parents[3] / 'data'
    db = OpenML_ASLC_Sqlite(path)
    df = db.load()

    # cols = list(df.columns[df.columns.str.startswith('test_accuracy_fold')])
    cols = [x for x in df.index.levels[0] if x.startswith('test_accuracy_fold')]
    cols.append('average_test_accuracy')
    tables = db.collect_tables(cols)

    # t = db.convert_to_tensor(list(cols))
    #
    # t.shape

    # df = tables['average_test_accuracy']
    # failed = set(df[df.isna().any(axis=1)].index)
    # log.debug('Failed dataset algorithm combinations: %s', failed)

    print('done')
