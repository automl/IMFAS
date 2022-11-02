
from pathlib import *
from omegaconf import DictConfig

import pdb

import gdown
import logging


log = logging.getLogger(__name__)

def get_taskset_data(url: str, save_path: str):
   
    x= gdown.download(url, save_path, quiet=False)



if __name__ == '__main__':
    url = 'https://drive.google.com/uc?id=1ILDw9hxO9qRcFzsbMBeTyVy6ddUmWQCa'
    path = Path(__file__).parents[3] / 'data' / 'preprocessed'/ 'task_set'
    output_name = 'raw.npy'

    get_taskset_data(url, str(path) + '/' + output_name)