from itertools import product

import numpy as np
import pandas as pd
from lcdb import get_all_curves, get_meta_features

# learning curves
df = get_all_curves(metric='accuracy')
df.sort_index(inplace=True)

threshold = 34  # upper bound on the power of sqrt2 for training set size.
inner, outer = 0, 0
size_test = 5000

sqrt2 = np.sqrt(2)
fidelity = pd.Series([int(sqrt2 ** k) for k in np.arange(start=8, stop=threshold, )])

# these two algorithms are removed, as they drastically reduce the number of available datasets.
removals = {'sklearn.naive_bayes.MultinomialNB',
            'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'}
learners = set(df.learner) - removals

# subset of df with frequent learners & fidelity of sqrt**k only!
df = df[df.learner.isin(learners)][df.size_train.isin(fidelity)]
df = df[(df.inner_seed == inner) & (df.outer_seed == outer) & (df.size_test == size_test)]
# fixme: remove size_test condition
df1 = df.groupby(['openmlid', 'size_train', ]).count()

# fixme: this does not consider the varying test size yet!
# fixing the test size to the most frequent (i.e. 5000) results in 16 Datasets.
df.size_test.value_counts()

# finding those datasets that have been observed at all fidelity levels
# FIXME: this is the most uggliest thing i've ever coded but it works.
algo_threshold = len(learners)
# algo_threshold = 17
a = df1.groupby(['openmlid']).agg(np.mean).learner >= algo_threshold
# available
openmlids = a.index[a].values

df = df[df.openmlid.isin(openmlids)]
df.set_index(['openmlid', 'learner'], inplace=True)

# find the set of learners per dataset in df
df.reset_index(inplace=True)
datasets, algos = set(df.openmlid), set(df.learner)
df.set_index(['openmlid', 'learner'], inplace=True)
avail_combinations = set(df.index)
all_combinations = set(product(datasets, algos))
missing = all_combinations - avail_combinations
print(f"Missing {len(missing) / len(all_combinations)} % dataset-algorithm combinations,\n"
      f"Missing the combinations {missing}\n\n")

print(f'Available datasets: {len(openmlids)}\n'
      f'Available learners: {len(learners)}\n'
      f'Available fidelity levels: {len(fidelity)}\n'
      f'Fidelity levels:\n {fidelity.values}')  # fixme: double check these

list(sorted(set(df.size_train)))

# dataset meta features
df_meta = get_meta_features()
df_meta = df_meta[df_meta.openmlid.isin(openmlids)]

# Obtain the learning curve tensor
# df = df[['size_train', 'score_test', 'score_train', 'score_valid']].reset_index()
#
# df.set_index('size_train').T
# -------------------------------------------------------------------------------------------
# from collections import Counter

# # count is n_learners with now max of 18!
# df2 = df.groupby(['openmlid', 'inner_seed']).count()
#
# # consider the frequency of datasets collected at size_test:
# # import matplotlib.pyplot as plt
# #
# # df.size_test.value_counts().sort_index().plot.bar()
# # plt.show()
#
# df.size_test.value_counts() > 50000
#
# # df1.size_train.value_counts().sort_index().plot.bar()
# # plt.show()
# # analysis on the fidelities occurences.
# outer, inner, size_test = 0, 0, 5000
#
# avail_seed_inner, avail_seed_outer, avail_size_test = set(df['inner_seed']), set(
#     df['outer_seed']), set(df['size_test'])
# df1 = df[(df['outer_seed'] == outer) & (df['inner_seed'] == inner) & (df['size_test'] == size_test)]
#
# datasets, algos = set(df1.openmlid), set(df1.learner)
# df1.set_index(['openmlid', 'learner'], inplace=True)
# avail_combinations = set(df1.index)
# all_combinations = set(product(datasets, algos))
# missing = all_combinations - avail_combinations
# print(f"Missing {len(missing) / len(all_combinations)} dataset-algorithm combinations")
#
# outer, inner = 1, 1
#
# df
# datasets, algos = set(df.openmlid), set(df.learner)
# df.set_index(['openmlid', 'learner'], inplace=True)
# avail_combinations = set(df.index)
# all_combinations = set(product(datasets, algos))
# missing = all_combinations - avail_combinations
# print(f"Missing {1 - len(missing) / len(all_combinations)} dataset-algorithm combinations")
#
# # TODO based on these subdataset frames, pick one seed
# df.set_index(['openmlid', 'learner'], inplace=True)
# curves = {(d, l): df[df.size_test == 5000].loc[(d, l)] for d, l in
#           set(df[df.size_test == 5000].index)}
#
# # fidelitiy analysis
# {(d, l): _df['size_train'] / _df['size_train'].max() for (d, l), _df in curves.items()}
#
# df1.groupby(['openmlid', 'learner'])['size_train'].transform(lambda x: x / x.max()).round(
#     decimals=2).value_counts().sort_index().plot.bar()
# plt.show()
#
# fid_frequency = df1.reset_index('learner').groupby(['learner', 'size_train']).count()
#
# learners = {l: df1.xs(l[1], level='learner') for l in set(df1.index)}
# # {l: [set(_df.index), ] for l, _df in learners}
#
# # containing all training, validation and test lc for each algo dataset combinations
# curves = {(d, l): df1.loc[(d, l)] for d, l in set(df1.index)}
# threshold = 5
# curves = {(d, l): _df for (d, l), _df in curves.items() if len(_df) > threshold}
# fidelities = {(d, l): set(_df.size_train) for (d, l), _df in curves.items()}
# set.intersection(*fidelities.values())
#
# fid_frequency.set_index(['learner', 'size_train', 'openmlid'], inplace=True)
# threshold = 11
# fid_frequency = fid_frequency[fid_frequency['openmlid'] > threshold]
#
# # d = dict()
# # for k, v in list(fid_frequency.index)
#
# fid_frequency.reset_index(inplace=True)
#
# # figure out the shared subset of algorithms on datasets
#
# # TODO: max-count for every dataset-algorithm combination
#
# df.set_index(['openmlid', 'learner'], inplace=True)
#
# Counter(df.index.get_level_values('openmlid'))
# Counter(df.index.get_level_values('learner'))
# Counter(df['size_train'])
#
# df[['size_train', 'size_test', 'inner_seed', 'outer_seed']].drop_duplicates()
#
# size_test = df.size_test.unique()
# size_test.sort()
# size_test
#
# size_train = df.size_train.unique()
# size_train.sort()
# size_train
#
# data_algo_combis = set(df.index)
# datasets = df.index.get_level_values('openmlid').unique()
# algorithms = df.index.get_level_values('learner').unique()
#
# outer, inner = 0, 0
# subset = df[(df['outer_seed'] == outer) & (df['inner_seed'] == inner)]
# subset.size_test.unique()
#
# excluded = [
#     (1483, 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'),
#     (40672, 'sklearn.ensemble.ExtraTreesClassifier'),
#     (4541, 'sklearn.neighbors.KNeighborsClassifier'),
#     (41228, 'sklearn.ensemble.ExtraTreesClassifier'),
#     (42769, 'sklearn.linear_model.LogisticRegression'),
#     (40668, 'sklearn.ensemble.GradientBoostingClassifier'),
#     (30, 'sklearn.ensemble.GradientBoostingClassifier'),
#     (201, 'SVC_sigmoid'),
#     (1236, 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'),
#     (1461, 'SVC_sigmoid'),
#     (26, 'SVC_rbf'),
#     (44, 'sklearn.linear_model.PassiveAggressiveClassifier'),
#     (4541, 'SVC_rbf'),
#     (26, 'sklearn.neural_network.MLPClassifier'),
#     (1235, 'sklearn.linear_model.LogisticRegression'),
#     (42733, 'sklearn.neighbors.KNeighborsClassifier'),
#     (1461, 'sklearn.ensemble.GradientBoostingClassifier'),
#     (180, 'sklearn.linear_model.RidgeClassifier'),
#     (184, 'sklearn.linear_model.LogisticRegression'),
#     (300, 'sklearn.tree.DecisionTreeClassifier'),
#     (42769, 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'),
#     (40672, 'SVC_rbf'),
#     (180, 'sklearn.naive_bayes.BernoulliNB'),
#     (40668, 'sklearn.ensemble.RandomForestClassifier'),
#     (180, 'sklearn.linear_model.SGDClassifier'),
#     (32, 'sklearn.neighbors.KNeighborsClassifier')
# ]
#
# # - {(d, neg) for d in datasets for neg in excluded}
# missing_keys = []
# data_algo_values = {}
# for idx in set(data_algo_combis):
#     try:
#         data_algo_values[idx] = subset.loc[idx]['size_train']
#
#     except KeyError:
#         missing_keys.append(idx)
#
# a = subset.size_train.value_counts().sort_index()
