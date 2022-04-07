from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
import glob
from io import StringIO
import pandas as pd

# COPY PASTE the table from HPOBENCH table 7
dataset_table = """
name tid #obs #feat
blood-transf 10101 748 4
vehicle 53 846 18
Australian 146818 690 14
car 146821 1728 6
phoneme 9952 5404 5
segment 146822 2310 19
credit-g 31 1000 20
kc1 3917 2109 22
sylvine 168912 5124 20
kr-vs-kp 3 3196 36
jungle_che 167119 44819 6
mfeat-factors 12 2000 216
shuttle 146212 58000 9
jasmine 168911 2984 145
cnae-9 9981 1080 856
numerai28.6 167120 96320 21
bank-mark 14965 45211 16
higgs 146606 98050 28
adult 7592 48842 14
nomao 9977 34465 118
"""

df = pd.read_csv(StringIO(dataset_table), sep=" ")

# installing once by selecting the model by name  ['lr', 'svm', 'xgb', 'rf', 'nn']
# then figure out the task_id: look at $HOME/.local/share/hpobench/TabularData/<model>.
# the displayed folders are the task ids!

# download the "tasks"
models = ['nn', 'rf', 'svm', 'xgb', 'lr']
for model in models:
    TabularBenchmark(model=model, task_id=146821)

# find available dataset ids
path = "/home/ruhkopf/.local/share/hpobench/TabularData/{model}/[0-9]*"
dataset_ids = {}
for model in models:
    s = set(int(path.split('/')[-1]) for path in glob.glob(path.format(model=model)))
    dataset_ids[model] = s

dataset_ids_nn = dataset_ids['nn']

# find the joint set of datasets used for all but the nn algo
dataset_ids = list(set.intersection(
    *[dataset_ids[model] for model in ['rf', 'svm', 'xgb', 'lr']]
))

model = 'svm'
dataset = dataset_ids[2]
seed = [8916, 1319, 7222, 7541, 665][0]
subsample = 1.0

tab = TabularBenchmark(model=model, task_id=dataset)
print('available fidelities: \n', tab.get_fidelity_space())

config = tab.get_configuration_space(seed=seed).sample_configuration()  # fixme: kill sample_config
result = tab.objective_function(configuration=config, fidelity={"subsample": subsample}, rng=seed)

# available seeds: (which are identical across datasets & algos
result['info'].keys() # [8916, 1319, 7222, 7541, 665]


for model in ['rf', 'svm', 'xgb', 'lr']:
    for dataset in dataset_ids:
        tab = TabularBenchmark(model=model, task_id=dataset)
        fid = tab.get_fidelity_space()
        print(f"Model {model}, {dataset}, subsample:{fid._hyperparameters['subsample'].sequence}")