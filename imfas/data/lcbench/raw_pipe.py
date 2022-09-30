import json
import logging
import pathlib

import hydra
import pandas as pd
from hydra.utils import call
from omegaconf import DictConfig

from imfas.data.lcbench.lcbench_api import LCBench_API
from imfas.data.util import subset

# A logger for this path
log = logging.getLogger(__name__)


def raw_pipe(*args, **kwargs):
    """
    Do heavy computation on the raw datasets - and move them to the data/preprocessed
    folder.

    For instance do Subset the HP-configurations by subsetting or ensembling.
    """

    # make it a hydra pipe again:
    cfg = DictConfig(kwargs)

    # directory paths
    # TODO @Tim change this to base config's data dir reference!
    orig_cwd = pathlib.Path(hydra.utils.get_original_cwd())
    dir_data = pathlib.Path(orig_cwd).parent / 'AlgoSelectionMF' / cfg.dir_data
    dir_downloads = dir_data / "downloads"
    dir_raw_dataset = dir_data / "raw" / "LCBench"

    dir_prep_dataset = dir_data / "preprocessed" / "LCBench"
    dir_prep_dataset.mkdir(parents=True, exist_ok=True)

    # todcheck if already downloaded the data
    if cfg.re_download:
        # TODO check me (and change the dir!)
        import subprocess

        # FIXME @Tim, can you take care of this?
        subprocess.call("~/PycharmProjects/AlgoSelectionMF/imfas/imfas/data/lcbench/download.sh")

    if cfg.reload_from_downloads:
        log.info("Starting to load jsons from path")
        # FIXME: move LCBench parsing into separate path (to make parsing dataset specific!
        # (0) get meta features
        with open(dir_downloads / cfg.dataset_name / "meta_features.json", "r") as file:
            df = pd.read_json(file, orient="index")

        df.to_csv(dir_data / "raw" / cfg.dataset_name / "meta_features.csv")

        log.info("Starting parsing.")
        # (1) parse the huge json into its components
        with open(dir_downloads / cfg.dataset_name / f"{cfg.extract}.json", "r") as file:
            DB = LCBench_API(json.load(file))
            # delattr(DB, data) # consider cleaning up after yourself to reduce memory burden!

        logs, config, results = DB.logs, DB.config, DB.results

        # write out relevant slices of this
        # check if dataset dir exists, else create
        pathlib.Path(dir_raw_dataset).mkdir(parents=True, exist_ok=True)

        log.info("Writing out parsed full sized h5-files")
        logs.to_hdf(dir_raw_dataset / "logs.h5", key="dataset", mode="w")
        results.to_hdf(dir_raw_dataset / "results.h5", key="dataset", mode="w")  # FIXME
        config.to_csv(dir_raw_dataset / "config.csv")

    else:
        log.info("Reading raw h5-files for subsetting them. ")
        # load files from raw dir
        logs = pd.read_hdf(dir_raw_dataset / "logs.h5", key="dataset")
        results = pd.read_hdf(dir_raw_dataset / "results.h5", key="dataset")
        config = pd.read_csv(dir_raw_dataset / "config.csv", index_col=0)
        meta_features = pd.read_csv(dir_raw_dataset / "meta_features.csv", index_col=0)

    # (0.1) final_performances
    # notice, that we could also get them as slices from logs!
    df = results[cfg.selection.lc_metric].unstack().T

    # (0.1.1) select based on final performance
    candidates, candidate_performances = call(cfg.selection.algo, df)

    candidates = list(sorted(candidates))

    # select the index rows # FIXME: this is inefficient
    config.index = config.index.astype(str)
    config = config.loc[candidates]
    # config.index = config.index.astype(int)
    # config = pd.DataFrame(
    #     [
    #         config.loc[element] if config.index.dtype == str
    #         else config.loc[element]
    #         for element in candidates
    #     ]
    # )
    # df.index = df.index.astype(str)

    # selecting the subset of algorithms!
    logs = subset(logs, "algorithm", candidates)
    results = subset(results, "algorithm", candidates)

    # subset(logs, 'logged', cfg.learning_curves.metrics)  # to select multiple metrics at once

    config.to_csv(dir_prep_dataset / "config_subset.csv")
    logs.to_hdf(dir_prep_dataset / "logs_subset.h5", key="dataset", mode="w")
    results.to_hdf(dir_prep_dataset / "results_subset.h5", key="dataset", mode="w")
    meta_features.to_csv(dir_prep_dataset / "meta_features.csv")
    log.debug("Written out all LCBench files to raw dir.")


# TODO rewrite download --> preprocessing folder
# TODO split raw_pipe into these parts
# TODO ensure that the download also works
# fixme: none of these actually need bench:
def collect_lctensor(algo_configs, bench, metric, fidelity_type, slices):
    pass


def collect_dataset_meta_features(bench):
    pass


def collect_algorithms(bench, cfg, fidelity_type):
    pass
