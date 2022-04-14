import json
import logging
import pathlib

import hydra
import pandas as pd
from hydra.utils import call
from omegaconf import DictConfig

from mf_gravitas.data.lcbench.lcbench_api import LCBench_API
from mf_gravitas.data.util import subset

# A logger for this file
log = logging.getLogger(__name__)


def main_raw(cfg: DictConfig):
    """
    Do heavy computation on the raw datasets - and move them to the data/preprocessing
    folder.

    For instance do Subset the HP-configurations by subsetting or ensembling.
    """
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    dir_data = pathlib.Path(cfg.dir_data)
    dir_downloads = dir_data / 'downloads'
    dir_raw_dataset = dir_data / 'raw' / cfg.dataset_name
    # todcheck if already downloaded the data
    if cfg.re_download:
        # TODO check me (and change the dir!)
        import subprocess
        subprocess.call(
            '~/PycharmProjects/AlgoSelectionMF/mf_gravitas/mf_gravitas/data/lcbench/download.sh')

    if cfg.reload_from_downloads:
        log.info('Starting to load jsons from file')
        # fixme: move LCBench parsing into separate file (to make parsing dataset specific!
        # (0) get meta features
        with open(dir_downloads / cfg.dataset_name / 'meta_features.json', 'r') as file:
            df = pd.read_json(file, orient='index')

        df.to_csv(dir_data / 'raw' / cfg.dataset_name / 'meta_features.csv')

        log.info('Starting parsing.')
        # (1) parse the huge json into its components
        with open(dir_downloads / cfg.dataset_name / f'{cfg.extract}.json', 'r') as file:
            DB = LCBench_API(json.load(file))
            # delattr(DB, data) # consider cleaning up after yourself to reduce memory burden!

        logs, config, results = DB.logs, DB.config, DB.results

        # write out relevant slices of this
        # check if dataset dir exists, else create
        pathlib.Path(dir_raw_dataset).mkdir(parents=True, exist_ok=True)

        log.info('Writing out parsed full sized h5-files')
        logs.to_hdf(dir_raw_dataset / 'logs.h5', key='dataset', mode='w')
        results.to_hdf(dir_raw_dataset / 'results.h5', key='dataset', mode='w')  # fixme
        config.to_csv(dir_raw_dataset / 'config.csv')

    else:
        log.info('Reading raw h5-files for subsetting them. ')
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
