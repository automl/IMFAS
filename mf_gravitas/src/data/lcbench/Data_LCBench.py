# TODO mkdir for other dataset.json optionally
# TODO optimize the paths used here (make the relative)
# TODO in both raw and preprocessed write to lcbench subfolder -
#  since there will be other datasets we'd like to compare against
import json
import logging

import pandas as pd

from api import Benchmark
from mf_gravitas.src.data.ensemble.topk import ensemble

logger = logging.getLogger(__name__)


class Data_LCBench:
    def __init__(self, rawdir, prepdir, metric: str = 'cross_entropy', ):
        """
           :param metric: any of the following ['accuracy', 'cross_entropy', 'balanced_accuracy']
           refers to the part of the available 'tags'
        """
        self.rawdir = rawdir
        self.prepdir = prepdir
        self.metric = metric
        self.dir = dir

        # TODO create data_3k folder (if it does not exist)
        # TODO read in the

        logger.info('Loading LCBench\'s huge json. This might take some time')
        self.bench = Benchmark('/home/ruhkopf/.local/share/LCBench/data_2k.json')
        self.names = self.bench.dataset_names
        self.tags = self.bench.get_queriable_tags()

    def select_subset(self, fnc, **kwargs):
        """
        Subsetting the data/raw/final_test_<self.metric>.csv file
        (algo X dataset final performances on test set measured in metric)
        to find suitably different algorithms.

        :param fnc: function, describing an algorithm that takes ase input the
        aforementioned df and returns a list of candidates  and the row subset dataframe of
        the aforementioned df accoring to the selection.
        :kwargs: kwargs to fnc.

        :example:
        from mf_gravitas.src.data.ensemble.topk import ensemble
        select_subset(ensemble, k=3)
        """
        df = pd.read_csv(f'{self.rawdir}/data_2k/final_test_{self.metric}.csv', index_col=0)

        candidates, candidate_performances = fnc(df, **kwargs)
        self.candidates = list(candidates)
        logger.debug(f'Ensembling algorithm {fnc} with kwargs {kwargs} found'
                     f' {len(self.candidates)} candidates.')

        candidate_performances.to_csv(f'{self.prepdir}/data_2k/final_test_{self.metric}.csv')

        # find train/validation final accuracy for subset
        for name in ['train', 'val']:
            df = pd.read_csv(f'{self.rawdir}/data_2k/final_{name}_{self.metric}.csv', index_col=0)
            df = df.iloc[self.candidates]
            df.to_csv(f'{self.prepdir}/data_2k/final_{name}_{self.metric}.csv')

    def read_configs(self):
        # find the candidate's configurations (algorithm_meta_features)
        configs = pd.read_csv(f'{self.rawdir}/data_2k/configs.csv', index_col=0)
        configs = configs.iloc[self.candidates]
        configs.to_csv(f'{self.prepdir}/data_2k/configs.csv')

    def _query_learning_curves(self, tag):
        """Convenience query wrapper for learning curves from LCBench.
         returns: Dict: key: tuple (dataset_name, algo_id) , value: list learning curve. """
        return {
            # string is required due to json.dump not accepting tuples
            str((dataset, algo)): self.bench.query(dataset_name=dataset, tag=tag, config_id=algo)
            for algo in self.candidates
            for dataset in self.names
        }

    def read_learning_curves(self):
        """
        Assuming the structure of LCBenchs' tracked metrics, here the subset of
        learning curves (train, validation & test) are extracted according to
        self._query_learning_curves and dumped into prepdir as separate jsons.
        """

        for typus in ['train', 'val', 'test']:
            tag = f'Train/{typus}_{self.metric}'

            # special case:
            if tag == 'Train/test_accuracy':
                tag = 'Train/test_result'

            curves = self._query_learning_curves(tag)

            # write out the curves into a json
            with open(f'{self.prepdir}/data_2k/{typus}_{self.metric}_curves.json', "w") as file:
                json.dump(curves, file)  # fixme: does not allow for tuple keys!

        loss = self._query_learning_curves('Train/loss')
        with open(f'{self.prepdir}/data_2k/train_loss.json', "w") as file:
            json.dump(loss, file)  # fixme: does not allow for tuple keys!


if __name__ == '__main__':
    lcbench = Data_LCBench(
        rawdir='/home/ruhkopf/PycharmProjects/AlgoSelectionMF/data/raw',
        prepdir='/home/ruhkopf/PycharmProjects/AlgoSelectionMF/data/preprocessed',
        metric='cross_entropy')

    lcbench.select_subset(ensemble, k=3)
    lcbench.read_configs()
    lcbench.read_learning_curves()
