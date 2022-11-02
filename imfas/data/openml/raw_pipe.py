import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from pandas import read_sql_query
import openml
import pdb 

log = logging.getLogger(__name__)


# FIXME: this is an API. still need to write the actual raw_pipe
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

        self.df = df

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

    def collect_dataset_meta_features(self):
        """Load the dataset meta features conditioned on the available datasets."""

        self.openmlids = self.df.index.levels[1]  # self.df.index.levels[0].values
        # add the missing values to the meta feature df
        datasets = []
        for openmlid in set(self.openmlids):
            datasets.append(openml.datasets.get_dataset(int(openmlid), download_data=False))

        qualities = {m.id: m.qualities for m in datasets}
        self.dataset_meta_features = pd.DataFrame.from_dict(qualities).T

    def save(self, cols):
        self.tables = dict()
        for col in cols:
            self.tables[col] = self.df.xs(key=col)

            # FIXME: this is preprocessing and does not belong here
            self.tables[col] = self.tables[col].loc[
                [k for k in self.tables[col].index
                 if k[1] not in set(['GaussianNB()', 'RadiusNeighborsClassifier()',
                                     'GaussianProcessClassifier()', 'NuSVC()'])]]
            self.tables[col].to_hdf(
                self.path_preprocessed / f"{col}_lcs.h5",
                key="dataset",
                mode="w"
            )

        self.dataset_meta_features.to_csv(self.path_preprocessed / f"dataset_meta_features.csv")

        return self.tables


def raw_pipe(**kwargs):
    cfg = DictConfig(kwargs)
    log.info('Loading from download')
    df = OpenML_ASLC_Sqlite(kwargs['root']).load_download(kwargs['path'])

    tables = db.save(cfg.cols)


if __name__ == '__main__':
    path = Path(__file__).parents[3] / 'data' / 'openml'
    
    db = OpenML_ASLC_Sqlite(path)

    df = db.load(from_download=True)
    
    print(df)
    pdb.set_trace()

    db.collect_dataset_meta_features()

    # cols = list(df.columns[df.columns.str.startswith('test_accuracy_fold')])
    cols = [x for x in df.index.levels[0] if x.startswith('test_accuracy_fold')]
    cols.append('average_test_accuracy')
    tables = db.save(cols)

    print('done')
