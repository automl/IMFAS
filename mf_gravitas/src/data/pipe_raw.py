import json
import logging
import pathlib

import hydra
import pandas as pd
from hydra.utils import call
from omegaconf import DictConfig

from .util import subset

# A logger for this file
log = logging.getLogger(__name__)


# FIXME: learning curves are read, ensembled based on Data_LCBench.py! move all
#  that stuff here!
class LCBench_API:
    def __init__(self, json):
        self.data = json
        self.names_datasets = list(self.data.keys())
        self.tags = self.get_queriable_tags(dataset_name='APSFailure')

        self.config, self.logs, self.results = self.parse()

    def query(self, dataset_name, tag, config_id):
        """
        Query a run.

        Keyword arguments:
        dataset_name -- str, the name of the dataset in the benchmark
        tag -- str, the tag you want to query
        config_id -- int, an identifier for which run you want to query, if too large will query the last run
        """
        config_id = str(config_id)
        if dataset_name not in self.names_datasets:
            raise ValueError("Dataset name not found.")

        if config_id not in self.data[dataset_name].keys():
            raise ValueError("Config nr %s not found for dataset %s." % (config_id, dataset_name))

        if tag in self.data[dataset_name][config_id]["log"].keys():
            return self.data[dataset_name][config_id]["log"][tag]

        if tag in self.data[dataset_name][config_id]["results"].keys():
            return self.data[dataset_name][config_id]["results"][tag]

        if tag in self.data[dataset_name][config_id]["config_raw"].keys():
            return self.data[dataset_name][config_id]["config_raw"][tag]

        if tag == "config_raw":
            return self.data[dataset_name][config_id]["config_raw"]

        raise ValueError(
            "Tag %s not found for config_raw %s for dataset %s" % (tag, config_id, dataset_name))

    def get_queriable_tags(self, dataset_name=None, config_id=None):
        """Returns a list of all queriable tags"""
        if dataset_name is None or config_id is None:
            dataset_name = list(self.data.keys())[0]
            config_id = list(self.data[dataset_name].keys())[0]
        else:
            config_id = str(config_id)
        log_tags = list(self.data[dataset_name][config_id]["log"].keys())
        result_tags = list(self.data[dataset_name][config_id]["results"].keys())
        config_tags = list(self.data[dataset_name][config_id]["config"].keys())
        return log_tags + result_tags + config_tags

    def parse(self):
        print()
        data_algo = {(d, a): data for d, algos in self.data.items() for a, data in algos.items()}
        names_algos = self.data[self.names_datasets[0]].keys()

        # logs = {(d, a): v for (d, a), v in data_algo.items()}
        logs = {(d, a, logged): v
                for (d, a), data in data_algo.items()
                for logged, v in data['log'].items()}

        configs = {k: data['config'] for k, data in data_algo.items()}
        results = {k: data['results'] for k, data in data_algo.items()}

        # parse as multi_index
        logs = pd.DataFrame.from_dict(logs)
        logs.columns.names = ['dataset', 'algorithm', 'logged']
        # change the default view for convenience
        logs = logs.T
        logs = logs.reorder_levels(['logged', 'dataset', 'algorithm'])

        # logs.loc['time']  # conveniently select the tracked feature
        # logged, datasets, algos = logs.index.levels # conveniently get the available options

        # Fixme: make this a debug flag and raise on difference in slices!
        if False:
            # to validate that across datasets the config is always the same
            # --> we only need one config per algorithm!
            config = pd.DataFrame.from_dict(configs)
            config.columns.names = ['dataset', 'algorithm']
            config.T.xs('1', level='algorithm')
            config.T.xs('2', level='algorithm')
            # config.T.xs(0, level='algorithm') # to extract a single algorithm

        config = {a: configs[(self.names_datasets[0], a)] for a in names_algos}
        config = pd.DataFrame.from_dict(config, orient='index')
        config.index.name = 'algorithm'

        results = pd.DataFrame.from_dict(results)
        results.columns.names = ['dataset', 'algorithm']
        results = results.T

        return config, logs, results


@hydra.main(config_path="config_raw", config_name="raw")
def main(cfg: DictConfig):
    """
    Do heavy computation on the raw datasets - and move them to the data/preprocessing
    folder.

    For instance do Subset the HP-configurations by subsetting or ensembling.
    """
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    dir_data = pathlib.Path(cfg.dir_data)
    dir_downloads = dir_data / 'downloads'
    dir_raw_dataset = dir_data / 'raw' / cfg.dataset
    # todcheck if already downloaded the data
    if cfg.re_download:
        # TODO check me (and change the dir!)
        import subprocess
        subprocess.call(
            '~/PycharmProjects/AlgoSelectionMF/mf_gravitas/src/data/lcbench/download.sh')

    if cfg.reload_from_downloads:
        log.info('Starting to load jsons from file')
        # fixme: move LCBench parsing into separate file (to make parsing dataset specific!
        # (0) get meta features
        with open(dir_downloads / cfg.dataset / 'meta_features.json', 'r') as file:
            df = pd.read_json(file, orient='index')

        with open(dir_data / 'raw' / cfg.extract / 'meta_features.csv', 'w') as file:
            df.to_csv(file)

        log.info('Starting parsing.')
        # (1) parse the huge json into its components
        with open(dir_downloads / cfg.dataset / f'{cfg.extract}.json', 'r') as file:
            DB = LCBench_API(json.load(file))
            # delattr(DB, data) # consider cleaning up after yourself to reduce memory burden!

        logs, config, results = DB.logs, DB.config, DB.results

        # write out relevant slices of this
        # check if dataset dir exists, else create
        pathlib.Path(dir_raw_dataset).mkdir(parents=True, exist_ok=True)

        logs.to_hdf(dir_raw_dataset / 'logs.h5', key='dataset', mode='w')
        results.to_hdf(dir_raw_dataset / 'results.h5', key='dataset', mode='w')  # fixme
        config.to_csv(dir_raw_dataset / 'config.csv')

    else:
        # load files from raw dir
        logs = pd.read_hdf(dir_raw_dataset / 'logs.h5', key='dataset')
        results = pd.read_hdf(dir_raw_dataset / 'results.h5', key='dataset')
        configs = pd.read_csv(dir_raw_dataset / 'config.csv')
        meta_features = pd.read_csv(dir_raw_dataset / 'meta_features.csv')

    # (0.1) final_performances
    # notice, that we could also get them as slices from logs!
    df = results[cfg.selection.metric].unstack().T

    # (0.1.1) select based on final performance
    candidates, candidate_performances = call(cfg.selection.algo, df)

    # selecting the subset of algorithms!
    logs = subset(logs, 'algorithm', list(candidates))
    results = subset(results, 'algorithm', list(candidates))

    subset(logs, 'logged', cfg.learning_curves.metrics)

    logs.to_hdf(dir_raw_dataset / 'logs_subset.h5', key='dataset', mode='w')
    results.to_hdf(dir_raw_dataset / 'results_subset.h5', key='dataset', mode='w')  # fixme

    log.debug('Written out all files to raw dir.')
