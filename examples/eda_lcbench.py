import glob

import pandas as pd

from api import Benchmark  # fixme LCBench.api

root = '/home/ruhkopf/.local/share/LCBench'
print(glob.glob(f"{root}/*.json"))

bench = Benchmark('/home/ruhkopf/.local/share/LCBench/fashion_mnist.json')
names = bench.dataset_names
tags = bench.get_queriable_tags()  # has probably the same as below

configs0 = {i: bench.query(dataset_name='Fashion-MNIST', tag="config_raw", config_id=i)
            for i in range(2000)}

configs_df0 = pd.DataFrame.from_dict(configs0, orient='index')

# the longer version with all the gradient information
# ['time', 'epoch', 'Train/loss', 'Train/train_accuracy', 'Train/val_accuracy',
# 'Train/train_cross_entropy', 'Train/val_cross_entropy',
# 'Train/train_balanced_accuracy', 'Train/val_balanced_accuracy', '
# Train/test_result', 'Train/test_cross_entropy', 'Train/test_balanced_accuracy',
# 'Train/gradient_max', 'Train/gradient_mean', 'Train/gradient_median',
# 'Train/gradient_std', 'Train/gradient_q10', 'Train/gradient_q25',
# 'Train/gradient_q75', 'Train/gradient_q90', 'Train/layer_wise_gradient_max_layer_0',
# 'Train/layer_wise_gradient_max_layer_1', 'Train/layer_wise_gradient_max_layer_2', '
# Train/layer_wise_gradient_mean_layer_0', 'Train/layer_wise_gradient_mean_layer_1',
# 'Train/layer_wise_gradient_mean_layer_2', 'Train/layer_wise_gradient_median_layer_0',
# 'Train/layer_wise_gradient_median_layer_1', 'Train/layer_wise_gradient_median_layer_2',
# 'Train/layer_wise_gradient_std_layer_0', 'Train/layer_wise_gradient_std_layer_1',
# 'Train/layer_wise_gradient_std_layer_2', 'Train/layer_wise_gradient_q10_layer_0',
# 'Train/layer_wise_gradient_q10_layer_1', 'Train/layer_wise_gradient_q10_layer_2',
# 'Train/layer_wise_gradient_q25_layer_0', 'Train/layer_wise_gradient_q25_layer_1',
# 'Train/layer_wise_gradient_q25_layer_2', 'Train/layer_wise_gradient_q75_layer_0',
# 'Train/layer_wise_gradient_q75_layer_1', 'Train/layer_wise_gradient_q75_layer_2',
# 'Train/layer_wise_gradient_q90_layer_0', 'Train/layer_wise_gradient_q90_layer_1',
# 'Train/layer_wise_gradient_q90_layer_2', 'Train/gradient_norm', 'Train/lr',
# 'model_parameters', 'final_train_cross_entropy', 'final_train_accuracy', '
# final_train_balanced_accuracy', 'final_val_cross_entropy', 'final_val_accuracy',
# 'final_val_balanced_accuracy', 'final_test_cross_entropy', 'final_test_accuracy',
# 'final_test_balanced_accuracy', 'OpenML_task_id', 'test_split', 'budget', 'seed',
# 'instances', 'classes', 'features', 'batch_size', 'imputation_strategy',
# 'learning_rate_scheduler', 'loss', 'network', 'max_dropout', 'normalization_strategy',
# 'optimizer', 'cosine_annealing_T_max', 'cosine_annealing_eta_min', 'activation',
# 'max_units', 'mlp_shape', 'num_layers', 'learning_rate', 'momentum', 'weight_decay', 'config_raw']
bench = Benchmark(data_dir='/home/ruhkopf/.local/share/LCBench/data_2k.json')

# the shorter version without all the gradient information
# ['time', 'epoch', 'Train/loss', 'Train/train_accuracy', 'Train/val_accuracy',
# 'Train/train_cross_entropy', 'Train/val_cross_entropy', 'Train/train_balanced_accuracy',
# 'Train/val_balanced_accuracy', 'Train/test_result', 'Train/test_cross_entropy',
# 'Train/test_balanced_accuracy', 'Train/lr', 'model_parameters', 'final_train_cross_entropy',
# 'final_train_accuracy', 'final_train_balanced_accuracy', 'final_val_cross_entropy',
# 'final_val_accuracy', 'final_val_balanced_accuracy', 'final_test_cross_entropy',
# 'final_test_accuracy', 'final_test_balanced_accuracy', 'OpenML_task_id',
# 'test_split', 'budget', 'seed', 'instances', 'classes', 'features', 'batch_size',
# 'imputation_strategy', 'learning_rate_scheduler', 'loss', 'network', 'max_dropout',
# 'normalization_strategy', 'optimizer', 'cosine_annealing_T_max', 'cosine_annealing_eta_min',
# 'activation', 'max_units', 'mlp_shape', 'num_layers', 'learning_rate', 'momentum',
# 'weight_decay', 'config_raw'])
bench = Benchmark(data_dir='/home/ruhkopf/.local/share/LCBench/data_2k_lw.json')
names = bench.dataset_names  # len: 35
tags = bench.get_queriable_tags()

# config_ids refers to the 2000 possible evaluated hp-configurations (0-1999)
bench.query(dataset_name='APSFailure', tag="config_raw", config_id=20)

# ! configs across datasets seem to be consistent!
dataset_name = 'APSFailure'

configs = {i: bench.query(dataset_name=dataset_name, tag="config_raw", config_id=i)
           for i in range(2000)}

configs_df = pd.DataFrame.from_dict(configs, orient='index')

dataset_name = 'Amazon_employee_access'
configs1 = {i: bench.query(dataset_name=dataset_name, tag="config_raw", config_id=i)
            for i in range(2000)}

configs_df1 = pd.DataFrame.from_dict(configs1, orient='index')
assert all(configs_df1 == configs_df)
assert all(configs_df0 == configs_df) # mnist also same configs.


# "full" dataset
bench = Benchmark('/home/ruhkopf/.local/share/LCBench/bench_full.json')
names = bench.dataset_names
tags = bench.get_queriable_tags()  # produces an error.


import json

with open('/home/ruhkopf/.local/share/LCBench/meta_features.json', "r") as read_file:
    data = json.load(read_file)

meta_features = pd.DataFrame.from_dict(data, orient='index')

null_share = meta_features.isnull().mean().round(4).mul(100).sort_values(ascending=False)

