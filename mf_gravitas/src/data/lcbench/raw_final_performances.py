"""
This script is intended to read out the various performance metrics & the configs
from a json file and write them to data/raw/<json base-name>
"""

import pandas as pd
import numpy as np
import os
from api import Benchmark




root = '/home/ruhkopf'
datadir = '/.local/share/LCBench'
dataraw = '/data/raw'
project = '/home/ruhkopf/PycharmProjects/AlgoSelectionMF'

with open(f'{root}/{datadir}/meta_features.json', 'r') as file:
    df = pd.read_json(file, orient='index')

df.to_csv(f'{dataraw}/meta_features.csv')


for file in ['data_2k_lw.json', 'fashion_mnist.json', 'data_2k.json']:


    file_raw_dir = f'{dataraw}/{file.split(".")[0]}'

    # create a folder for the derivates of this particular json file
    os.mkdir(f'{project}/{file_raw_dir}')
    bench = Benchmark(f'{root}/{datadir}/{file}')
    names = bench.dataset_names
    tags = bench.get_queriable_tags()

    # create a csv file for each final performance of this particular json instance
    for tag in (tag for tag in tags if tag.startswith('final')):
        tmp = np.zeros((2000, len(names)))
        for n, name in enumerate(names):
            for c in range(2000):
                tmp[c, n] = bench.query(dataset_name=name, tag=tag, config_id=c)

        pd.DataFrame(tmp, columns=names).to_csv(f'{project}/{file_raw_dir}/{tag}.csv')

    # create a csv with all the configs
    # validated, that all configs are the same across datasets:
    configs = {i: bench.query(dataset_name=names[0], tag="config", config_id=i)
               for i in range(2000)}
    configs_df = pd.DataFrame.from_dict(configs, orient='index')


    # check if configs are the same all across datasets
    try:
        for i, name in enumerate(names):
            configs1 = {i: bench.query(dataset_name=name, tag="config", config_id=i)
                        for i in range(2000)}
            configs_df1 = pd.DataFrame.from_dict(configs1, orient='index')
            assert all(configs_df == configs_df1)

        # write out the config to file
        configs_df.to_csv(f'{project}/{file_raw_dir}/configs.csv')

    except AssertionError:
        print('configs do not match across datasets')




