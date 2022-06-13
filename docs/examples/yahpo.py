"""
YAHPO
-----

.. warning::
    This page is under construction.
    
"""


# pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"

"""
openml_ids = {
    'lcbench': [168868, 167152, 167200, 168330, 189862, 189909, 167181, 167149, 126026, 168910,
                167083, 167190, \
                189906, 146212, 189865, 167168, 168329, 189873, 34539, 189866, 167104, 189908, 3945,
                168335,
                126025, 189354, 126029, 167161, 167184, 189905, 168908, 167185, 167201, 7593,
                168331],

    'rbv2_svm': [40978, 40966, 42, 377, 458, 307, 1515, 1510, 16, 334, 50, 40984, 40975, 1479, 54,
                 15, 12, 1468, 40496, \
                 1501, 188, 40994, 1462, 1063, 40979, 469, 22, 23381, 29, 312, 6332, 40981, 470,
                 40499, 1480, 18,
                 14, 11, 1494, 1049, 37, 41157, 31, 1068, 38, 40670, 23, 40982, 40900, 181, 1050,
                 1464, 40983, 28, 1487,
                 46, 1067, 3, 41212, 1056, 1497, 182, 375, 32, 41143, 4154, 60, 24, 44, 41146,
                 40498, 41156, 1475, 40701, 1485, 41216, 1489, 4135, 1040, 4534, 4538, 1476, 1457,
                 1053, 4134, 300, 1478, 40536, 1111, 1461, 41142, 1220, 41163, 41138, 1486, 41164],

    'rbv2_ranger': [41157, 40984, 54, 1464, 181, 40496, 1494, 1050, 40975, 40994, 15, 1468, 1462,
                    1063, 1510, 334, 377, \
                    11, 1068, 18, 40982, 37, 469, 1487, 23, 1515, 50, 458, 1480, 1067, 40966, 40983,
                    1049, 12,
                    1497, 40979, 31, 42, 182, 40900, 1056, 28, 1501, 40498, 16, 1479, 14, 60, 40499,
                    44, 41146, 41156,
                    4154, 40981, 22, 32, 1489, 1040, 3, 41216, 38, 312, 46, 29, 375, 1475, 40685,
                    1485, 6332, 470, 40701, 41143, 41278, 41027, 24, 1457, 307, 40670, 188, 4538, 6,
                    23381, 1053, 41163, 41165, 4134, 41212, 1478, 40536, 41142, 1476, 41161, 4534,
                    41159, 40978, 41164, 300, 1486, 1111, 1220, 151, 41138, 40927, 4135, 41166,
                    1590, 4541, 1461, 40668, 23517, 41162, 23512, 41168, 41150],

    'rbv2_rpart': [1111, 1486, 41164, 1478, 182, 40979, 40670, 40536, 300, 4538, 4154, 1040, 41138,
                   41212, 1468, 375, \
                   1476, 1485, 4134, 22, 4534, 1068, 1479, 40978, 31, 470, 11, 37, 1475, 16, 24,
                   377, 18, 40966,
                   40984, 23381, 1497, 40975, 1056, 458, 1501, 1063, 41162, 41142, 40900, 307, 15,
                   41156, 14, 3, 1494,
                   1067, 181, 44, 41163, 40983, 38, 42, 40994, 4135, 312, 1053, 29, 1510, 40701, 60,
                   1462, 1515, 1487, 50, 41146, 6332, 40498, 41143, 1489, 54, 46, 23, 40982, 32, 28,
                   1049, 1464, 12, 40499, 1050, 1590, 41161, 1457, 40981, 41159, 334, 1480, 41165,
                   40496, 469, 40927, 41157, 188, 23512, 1461],

    'rbv2_glmnet': [41156, 38, 40701, 60, 24, 469, 37, 23381, 15, 1464, 29, 40981, 11, 4534, 40978,
                    40994, 458, 312, 1063, \
                    1489, 3, 1068, 1510, 18, 6332, 23, 41143, 470, 1462, 4538, 44, 1067, 1053, 31,
                    1497, 41146,
                    40983, 1494, 41212, 1040, 50, 41157, 40984, 40536, 40900, 1050, 54, 1485, 1487,
                    40496, 1468,
                    40975, 1049, 1480, 181, 375, 40982, 1475, 1461, 4154, 334, 377, 1590, 40966,
                    1515, 16, 40979, 1056, 22, 182, 188, 1501, 40498, 4541, 12, 14, 40499, 1486, 42,
                    32, 23512, 28, 307, 1479, 46, 41278, 4134, 40670, 41162, 4135, 41159, 40668,
                    41161, 41142, 1111, 41138, 1478, 1476],

    'rbv2_xgboost': [3, 54, 38, 41278, 41156, 42, 60, 41161, 375, 4154, 41216, 41162, 32, 1468, 46,
                     1479, 50, 1497, 40982, \
                     1461, 41143, 41146, 40499, 300, 4534, 6332, 4135, 14, 41157, 12, 22, 40496,
                     1478, 29, 40978, 312,
                     40670, 469, 1476, 1487, 4134, 1590, 15, 40994, 41212, 188, 1501, 470, 377,
                     1515, 181, 1486, 18,
                     458, 40975, 40498, 1050, 41142, 1489, 4541, 40979, 1485, 23, 1040, 24, 1480,
                     40701, 40966, 41159, 1049, 44, 1068, 40900, 1464, 31, 4538, 23381, 40981, 1067,
                     1510, 182, 37, 40983, 307, 1475, 1494, 16, 41163, 11, 1111, 334, 1063, 1053,
                     41164, 40984, 1056, 41138, 40536, 1462, 28, 40668, 41165, 1457, 1220, 41150,
                     40927, 23512, 151, 41166],

    'rbv2_aknn': [469, 181, 40496, 1464, 1462, 11, 334, 40981, 42, 1480, 18, 40994, 1063, 1068,
                  1510, 15, 54, 50, 23381, \
                  307, 23, 37, 29, 470, 40975, 188, 31, 377, 6332, 22, 16, 14, 182, 375, 1501, 1515,
                  1475, 60, 1497,
                  4538, 12, 40979, 40499, 300, 28, 1479, 1053, 32, 41143, 1468, 312, 41212, 458,
                  1476, 1494, 40984,
                  1049, 4134, 4534, 1478, 1050, 40966, 40982, 41156, 1067, 1485, 40900, 40498, 1487,
                  1489, 40983, 46, 40536, 1056, 40670, 38, 44, 41146, 40701, 3, 1457, 1040, 41142,
                  1220, 41164, 4154, 41278, 24, 1486, 41163, 40978, 41138, 41157, 1111, 41159,
                  41162, 41161, 41165, 1461],

    'rbv2_super': [42, 377, 40966, 1510, 458, 54, 334, 40975, 15, 1462, 50, 1515, 40496, 40994, 469,
                   40984, 1063, 40978, \
                   307, 16, 1468, 11, 18, 40979, 12, 1479, 1501, 37, 1480, 1464, 1068, 40981, 181,
                   22, 29, 1494, 23381,
                   31, 23, 188, 470, 312, 14, 6332, 1049, 40499, 41157, 1050, 40982, 1487, 40900,
                   40983, 38, 1067, 28,
                   1497, 182, 3, 40670, 60, 1056, 46, 375, 44, 41156, 41212, 4154, 41146, 32, 41143,
                   24, 1489, 40498, 1475, 40701, 1040, 1485, 4538, 4534, 1053, 40536, 4134, 1478,
                   1476, 1486, 41142, 1111, 41138, 1461],

    'iaml_ranger': [1489, 41146, 40981, 1067],

    'iaml_rpart': [41146, 1489, 40981, 1067],

    'iaml_glmnet': [40981, 41146, 1489, 1067],

    'iaml_xgboost': [41146, 1489, 1067, 40981],

    'iaml_super': [41146, 1489, 40981, 1067]
}

# none of the datasets are used in all benchmarks!
set.intersection({k: set(v) for k, v in openml_ids.items()}.values())  # rbv2 & iaml intersection!

set.intersection(*[set(v) for k, v in openml_ids.items() if k != 'lcbench'])

iaml = set.intersection(*[set(v) for k, v in openml_ids.items() if k.startswith('iaml')])
rbv2 = set.intersection(*[set(v) for k, v in openml_ids.items() if k.startswith('rbv2')])
rbv2_k = [k for k in openml_ids.keys() if k.startswith('rbv2')]
iaml_k = [k for k in openml_ids.keys() if k.startswith('iaml')]

from openml.datasets import get_dataset
from openml.tasks import get_tasks, get_task
from yahpo_gym import *

bench = BenchmarkSet(scenario="lcbench")
bench.instances
bench.set_instance(bench.instances[0])

config = bench.config_space.sample_configuration(1).get_dictionary()
fidelity_name = 'epoch'
config[fidelity_name]
# Evaluate
print(bench.objective_function(config))

tasks = get_tasks(openml_ids['lcbench'], download_data=False)
task = get_task(openml_ids['lcbench'][0], download_data=False)
dataset = get_dataset(dataset_id=task.dataset_id, download_data=False)

import pandas as pd

dataset_meta_features = pd.DataFrame.from_dict({task.dataset_id: dataset.qualities}).T

# TODO Algorithm meta features


# Sample yahpo using todo latin hypercube / sobol design -- make this optional!
# depending on the number of sobol, it might make sense to use the todo  topk sampling strategy,
# which we already have.
smac.initial_design.sobol_design
smac.initial_design.latin_hypercube_design.LHDesign

# TODO always the best performing config (raw/yahpo_data/<bench>/best_params_resnet.json)

# todo configspace is here : (raw/yahpo_data/<bench>/config_space.json) (has to be refactored
#  though!

# TODO: choose either noisy or not noisy!

configs = LHDesign(
    cs: ConfigSpace.configuration_space.ConfigurationSpace,
        rng: numpy.random.mtrand.RandomState = None,
                                               traj_logger: smac.utils.io.traj_logging.TrajLogger = None,
                                                                                                    ta_run_limit: int = 99999,
                                                                                                                        configs:
Optional[List[ConfigSpace.configuration_space.Configuration]] = None,
                                                                n_configs_x_params: Optional[
    int] = 10,
)

# Write out to
"""