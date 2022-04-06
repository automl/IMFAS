import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns


class Dataset_Gravity(Dataset):
    # Dataset_columns
    columns_descriptive = ["usage", "name"]
    columns_categorical = ["task", "target_type", "feat_type", "metric"]
    columns_binary = ['has_categorical', 'has_missing', 'is_sparse']
    columns_numerical = ['time_budget', 'feat_num', 'target_num', 'label_num', 'train_num', 'valid_num', 'test_num']

    # Encoder must be available across instances
    enc_cat = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_num = MinMaxScaler()

    # ensure that we will not ignore any observations for categorical variables
    enc_cat.fit(
        pd.concat([
            pd.DataFrame.from_dict(data={'task': {'carlo': 'binary.classification',
                                                  'christine': 'binary.classification',
                                                  'digits': 'multiclass.classification',
                                                  'dilbert': 'multiclass.classification',
                                                  'dorothea': 'binary.classification',
                                                  'evita': 'binary.classification',
                                                  'flora': 'regression',
                                                  'grigoris': 'multilabel.classification',
                                                  'newsgroups': 'multiclass.classification',
                                                  'robert': 'multiclass.classification',
                                                  'tania': 'multilabel.classification',
                                                  'waldo': 'multiclass.classification',
                                                  'wallis': 'multiclass.classification',
                                                  'yolanda': 'regression',
                                                  'adult': 'multilabel.classification',
                                                  'albert': 'binary.classification',
                                                  'alexis': 'multilabel.classification',
                                                  'arturo': 'multiclass.classification'},
                                         'target_type': {'carlo': 'Binary',
                                                         'christine': 'Binary',
                                                         'digits': 'Categorical',
                                                         'dilbert': 'Categorical',
                                                         'dorothea': 'Binary',
                                                         'evita': 'Categorical',
                                                         'flora': 'Numerical',
                                                         'grigoris': 'Categorical',
                                                         'newsgroups': 'Numerical',
                                                         'robert': 'Binary',
                                                         'tania': 'Binary',
                                                         'waldo': 'Categorical',
                                                         'wallis': 'Categorical',
                                                         'yolanda': 'Numerical',
                                                         'adult': 'Binary',
                                                         'albert': 'Categorical',
                                                         'alexis': 'Binary',
                                                         'arturo': 'Categorical'},
                                         'feat_type': {'carlo': 'Numerical',
                                                       'christine': 'Numerical',
                                                       'digits': 'Numerical',
                                                       'dilbert': 'Numerical',
                                                       'dorothea': 'Binary',
                                                       'evita': 'Numerical',
                                                       'flora': 'Numerical',
                                                       'grigoris': 'Numerical',
                                                       'newsgroups': 'Numerical',
                                                       'robert': 'Numerical',
                                                       'tania': 'Numerical',
                                                       'waldo': 'Mixed',
                                                       'wallis': 'Numerical',
                                                       'yolanda': 'Numerical',
                                                       'adult': 'Mixed',
                                                       'albert': 'Numerical',
                                                       'alexis': 'Numerical',
                                                       'arturo': 'Numerical'},
                                         'metric': {'carlo': 'pac_metric',
                                                    'christine': 'bac_metric',
                                                    'digits': 'bac_metric',
                                                    'dilbert': 'pac_metric',
                                                    'dorothea': 'auc_metric',
                                                    'evita': 'auc_metric',
                                                    'flora': 'a_metric',
                                                    'grigoris': 'auc_metric',
                                                    'newsgroups': 'pac_metric',
                                                    'robert': 'bac_metric',
                                                    'tania': 'pac_metric',
                                                    'waldo': 'bac_metric',
                                                    'wallis': 'auc_metric',
                                                    'yolanda': 'r2_metric',
                                                    'adult': 'f1_metric',
                                                    'albert': 'f1_metric',
                                                    'alexis': 'auc_metric',
                                                    'arturo': 'f1_metric'}}),
            pd.DataFrame.from_dict({'task': {'24': 'multilabel.classification',
                                             '25': 'multiclass.classification',
                                             '26': 'multilabel.classification',
                                             '27': 'binary.classification',
                                             '28': 'multilabel.classification',
                                             '29': 'multilabel.classification',
                                             '3': 'regression',
                                             '30': 'multiclass.classification',
                                             '31': 'multilabel.classification',
                                             '32': 'multilabel.classification',
                                             '33': 'binary.classification',
                                             '34': 'multiclass.classification',
                                             '35': 'multiclass.classification',
                                             '36': 'regression',
                                             '37': 'multilabel.classification',
                                             '38': 'regression',
                                             '39': 'multilabel.classification',
                                             '4': 'multilabel.classification',
                                             '41': 'multilabel.classification',
                                             '42': 'binary.classification',
                                             '43': 'binary.classification',
                                             '44': 'binary.classification',
                                             '45': 'multilabel.classification',
                                             '46': 'binary.classification',
                                             '47': 'binary.classification',
                                             '48': 'multiclass.classification',
                                             '5': 'multilabel.classification',
                                             '50': 'binary.classification',
                                             '51': 'multiclass.classification',
                                             '52': 'regression',
                                             '53': 'multiclass.classification',
                                             '54': 'multiclass.classification',
                                             '55': 'regression',
                                             '57': 'multilabel.classification',
                                             '58': 'multilabel.classification',
                                             '59': 'binary.classification',
                                             '6': 'binary.classification',
                                             '60': 'binary.classification',
                                             '63': 'multiclass.classification',
                                             '64': 'multilabel.classification',
                                             '65': 'regression',
                                             '66': 'multilabel.classification',
                                             '67': 'multiclass.classification',
                                             '68': 'binary.classification',
                                             '69': 'regression',
                                             '7': 'binary.classification',
                                             '70': 'multilabel.classification',
                                             '71': 'regression',
                                             '72': 'multiclass.classification',
                                             '73': 'multiclass.classification',
                                             '74': 'multilabel.classification',
                                             '75': 'multiclass.classification',
                                             '76': 'binary.classification',
                                             '77': 'multiclass.classification',
                                             '78': 'multiclass.classification',
                                             '79': 'multilabel.classification',
                                             '80': 'multilabel.classification',
                                             '81': 'multilabel.classification',
                                             '82': 'multilabel.classification',
                                             '85': 'multilabel.classification',
                                             '86': 'multiclass.classification',
                                             '87': 'multiclass.classification',
                                             '88': 'regression',
                                             '89': 'regression',
                                             '9': 'binary.classification',
                                             '90': 'binary.classification',
                                             '91': 'regression',
                                             '92': 'multilabel.classification',
                                             '93': 'binary.classification',
                                             '94': 'binary.classification',
                                             '95': 'binary.classification',
                                             '96': 'regression',
                                             '97': 'multiclass.classification',
                                             '98': 'binary.classification',
                                             '99': 'regression'},
                                    'target_type': {'24': 'Numerical',
                                                    '25': 'Binary',
                                                    '26': 'Categorical',
                                                    '27': 'Categorical',
                                                    '28': 'Numerical',
                                                    '29': 'Numerical',
                                                    '3': 'Categorical',
                                                    '30': 'Categorical',
                                                    '31': 'Numerical',
                                                    '32': 'Numerical',
                                                    '33': 'Binary',
                                                    '34': 'Numerical',
                                                    '35': 'Categorical',
                                                    '36': 'Categorical',
                                                    '37': 'Numerical',
                                                    '38': 'Binary',
                                                    '39': 'Numerical',
                                                    '4': 'Categorical',
                                                    '41': 'Binary',
                                                    '42': 'Categorical',
                                                    '43': 'Binary',
                                                    '44': 'Numerical',
                                                    '45': 'Categorical',
                                                    '46': 'Categorical',
                                                    '47': 'Binary',
                                                    '48': 'Numerical',
                                                    '5': 'Binary',
                                                    '50': 'Numerical',
                                                    '51': 'Binary',
                                                    '52': 'Binary',
                                                    '53': 'Categorical',
                                                    '54': 'Binary',
                                                    '55': 'Numerical',
                                                    '57': 'Categorical',
                                                    '58': 'Categorical',
                                                    '59': 'Numerical',
                                                    '6': 'Categorical',
                                                    '60': 'Binary',
                                                    '63': 'Categorical',
                                                    '64': 'Binary',
                                                    '65': 'Binary',
                                                    '66': 'Binary',
                                                    '67': 'Binary',
                                                    '68': 'Binary',
                                                    '69': 'Categorical',
                                                    '7': 'Numerical',
                                                    '70': 'Binary',
                                                    '71': 'Categorical',
                                                    '72': 'Categorical',
                                                    '73': 'Numerical',
                                                    '74': 'Numerical',
                                                    '75': 'Binary',
                                                    '76': 'Numerical',
                                                    '77': 'Binary',
                                                    '78': 'Categorical',
                                                    '79': 'Numerical',
                                                    '80': 'Numerical',
                                                    '81': 'Categorical',
                                                    '82': 'Binary',
                                                    '85': 'Categorical',
                                                    '86': 'Categorical',
                                                    '87': 'Categorical',
                                                    '88': 'Binary',
                                                    '89': 'Categorical',
                                                    '9': 'Binary',
                                                    '90': 'Numerical',
                                                    '91': 'Categorical',
                                                    '92': 'Categorical',
                                                    '93': 'Categorical',
                                                    '94': 'Numerical',
                                                    '95': 'Numerical',
                                                    '96': 'Categorical',
                                                    '97': 'Numerical',
                                                    '98': 'Numerical',
                                                    '99': 'Numerical'},
                                    'feat_type': {'24': 'Numerical',
                                                  '25': 'Categorical',
                                                  '26': 'Binary',
                                                  '27': 'Binary',
                                                  '28': 'Binary',
                                                  '29': 'Binary',
                                                  '3': 'Binary',
                                                  '30': 'Numerical',
                                                  '31': 'Categorical',
                                                  '32': 'Numerical',
                                                  '33': 'Categorical',
                                                  '34': 'Categorical',
                                                  '35': 'Categorical',
                                                  '36': 'Categorical',
                                                  '37': 'Binary',
                                                  '38': 'Categorical',
                                                  '39': 'Categorical',
                                                  '4': 'Binary',
                                                  '41': 'Binary',
                                                  '42': 'Mixed',
                                                  '43': 'Mixed',
                                                  '44': 'Mixed',
                                                  '45': 'Binary',
                                                  '46': 'Categorical',
                                                  '47': 'Categorical',
                                                  '48': 'Categorical',
                                                  '5': 'Numerical',
                                                  '50': 'Mixed',
                                                  '51': 'Binary',
                                                  '52': 'Binary',
                                                  '53': 'Numerical',
                                                  '54': 'Binary',
                                                  '55': 'Numerical',
                                                  '57': 'Numerical',
                                                  '58': 'Numerical',
                                                  '59': 'Numerical',
                                                  '6': 'Binary',
                                                  '60': 'Mixed',
                                                  '63': 'Categorical',
                                                  '64': 'Categorical',
                                                  '65': 'Mixed',
                                                  '66': 'Binary',
                                                  '67': 'Mixed',
                                                  '68': 'Mixed',
                                                  '69': 'Numerical',
                                                  '7': 'Numerical',
                                                  '70': 'Numerical',
                                                  '71': 'Numerical',
                                                  '72': 'Categorical',
                                                  '73': 'Categorical',
                                                  '74': 'Mixed',
                                                  '75': 'Numerical',
                                                  '76': 'Binary',
                                                  '77': 'Binary',
                                                  '78': 'Binary',
                                                  '79': 'Binary',
                                                  '80': 'Mixed',
                                                  '81': 'Mixed',
                                                  '82': 'Mixed',
                                                  '85': 'Categorical',
                                                  '86': 'Mixed',
                                                  '87': 'Categorical',
                                                  '88': 'Categorical',
                                                  '89': 'Numerical',
                                                  '9': 'Numerical',
                                                  '90': 'Numerical',
                                                  '91': 'Binary',
                                                  '92': 'Mixed',
                                                  '93': 'Mixed',
                                                  '94': 'Numerical',
                                                  '95': 'Categorical',
                                                  '96': 'Binary',
                                                  '97': 'Binary',
                                                  '98': 'Binary',
                                                  '99': 'Numerical'},
                                    'metric': {'24': 'auc_metric',
                                               '25': 'auc_metric',
                                               '26': 'f1_metric',
                                               '27': 'r2_metric',
                                               '28': 'f1_metric',
                                               '29': 'a_metric',
                                               '3': 'auc_metric',
                                               '30': 'r2_metric',
                                               '31': 'bac_metric',
                                               '32': 'r2_metric',
                                               '33': 'pac_metric',
                                               '34': 'bac_metric',
                                               '35': 'bac_metric',
                                               '36': 'a_metric',
                                               '37': 'auc_metric',
                                               '38': 'bac_metric',
                                               '39': 'auc_metric',
                                               '4': 'pac_metric',
                                               '41': 'f1_metric',
                                               '42': 'f1_metric',
                                               '43': 'a_metric',
                                               '44': 'a_metric',
                                               '45': 'r2_metric',
                                               '46': 'bac_metric',
                                               '47': 'f1_metric',
                                               '48': 'f1_metric',
                                               '5': 'r2_metric',
                                               '50': 'f1_metric',
                                               '51': 'f1_metric',
                                               '52': 'bac_metric',
                                               '53': 'bac_metric',
                                               '54': 'auc_metric',
                                               '55': 'a_metric',
                                               '57': 'f1_metric',
                                               '58': 'r2_metric',
                                               '59': 'auc_metric',
                                               '6': 'auc_metric',
                                               '60': 'f1_metric',
                                               '63': 'pac_metric',
                                               '64': 'f1_metric',
                                               '65': 'auc_metric',
                                               '66': 'bac_metric',
                                               '67': 'bac_metric',
                                               '68': 'bac_metric',
                                               '69': 'r2_metric',
                                               '7': 'bac_metric',
                                               '70': 'f1_metric',
                                               '71': 'f1_metric',
                                               '72': 'f1_metric',
                                               '73': 'r2_metric',
                                               '74': 'r2_metric',
                                               '75': 'f1_metric',
                                               '76': 'f1_metric',
                                               '77': 'bac_metric',
                                               '78': 'f1_metric',
                                               '79': 'bac_metric',
                                               '80': 'auc_metric',
                                               '81': 'a_metric',
                                               '82': 'r2_metric',
                                               '85': 'a_metric',
                                               '86': 'f1_metric',
                                               '87': 'auc_metric',
                                               '88': 'pac_metric',
                                               '89': 'auc_metric',
                                               '9': 'bac_metric',
                                               '90': 'r2_metric',
                                               '91': 'r2_metric',
                                               '92': 'bac_metric',
                                               '93': 'auc_metric',
                                               '94': 'f1_metric',
                                               '95': 'auc_metric',
                                               '96': 'a_metric',
                                               '97': 'auc_metric',
                                               '98': 'pac_metric',
                                               '99': 'pac_metric'}})
        ]))

    n_features = 0

    def __init__(self, dataset_meta_features, learning_curves, algorithms_meta_features,
                 no_competitors=11,
                 deselect=0, topk=10, deselection_metric='skew', seed=123456, ):
        """

        :param dataset_meta_features:
        :param learning_curves:
        :param algorithms_meta_features:
        :param no_competitors: number of datasets that are compared against. Notice that no_competitors = 2
        :param deselect: int number of algorithms to deselect using backward selection
        :param topk: int in deselection: number of topk performing algorithms to consider for selection
        :param deselection_metric: str. 'skew' or '0threshold'. Skewness based selection looks at how the skewness
        of the change in performance of topk across all datasets is improved towards improving the avg topk performance.
        0threshold removes the algorithm with minimal density for decreased avg-topk performance across datasets (i.e.
        the integral from -inf to 0 of that ecdf.
        is pairwise comparisons
        """
        self.no_competitors = no_competitors
        self.deselect = deselect
        self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

        if deselect > 0:
            self.deselected = self._reduce_algo_space(removals=deselect, k=topk, mode=deselection_metric)
            # redoit all again, so that the getitem never sees these algos
            self.preprocess_with_known_deselection(self.deselected, dataset_meta_features, learning_curves,
                                                   algorithms_meta_features)

        self.nA = len(self.algo_final_performances.columns)

        # needed for plotting
        self.raw_learning_curves = learning_curves
        self.raw_dataset_meta_features = dataset_meta_features

        # seeding
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return len(self.datasets_meta_features)

    def __getitem__(self, item):
        """
        Compareset get logic:
        :param item: int. index in the range(0, len(self)) of a dataset that is to be optimized.
        :return: D0, D1, A0, A1
        D0: meta features of the dataset that we want to optimize
        A0: algo performances for D0
        D1: set of datasets' meta features to compare against (of length k)
        A1: respective algo performances for D1
        """
        # get the dataset & its performances
        D0 = self.datasets_meta_features[item]
        A0 = self.algo_thresholded_performances[item]

        # generate a random compare set
        # TODO move k to init
        # TODO seedit
        item_compareset = random.choices(list(set(range(self.nD)) - {item}), k=self.no_competitors)
        D1 = self.datasets_meta_features[item_compareset]
        A1 = self.algo_thresholded_performances[item_compareset]

        return D0, D1, A0, A1

    def preprocess(
            self,
            dataset_meta_features,
            learning_curves,
            algorithms_meta_features,
            k=3,
    ):
        """
        (1) Dataset Meta Features are selected based on being variable and transformed to tensor + normalized
        (2) Learning Curves: for each LC particular properties are read out such as e.g. final performance.
        (3) Algorithm Features: LC properties aggregated by algorithm
        (4) Dataset Features LC properties aggregated by dataset

        :param dataset_meta_features:
        :param learning_curves:
        :param algorithms_meta_features:
        :param k: (optional) set the k least performing algorithm performances to 0
        :return:
        """
        self.nD = len(dataset_meta_features.keys())

        # changing keys to int
        algorithms_meta_features = {k: v for k, v in algorithms_meta_features.items()}
        dataset_meta_features = {k: v for k, v in dataset_meta_features.items()}
        learning_curves = {k: {int(k1): v1 for k1, v1 in v.items()}
                           for k, v in learning_curves.items()}

        self._preprocess_meta_features(dataset_meta_features)
        self._preprocess_learning_curves(algorithms_meta_features, learning_curves)
        self._preporcess_scalar_properties(self.algo_learning_curves)
        self._preprocess_dataset_properties(learning_curves, dataset_meta_features)
        self._preprocess_thresholded_algo_performances(k=10)

    def preprocess_with_known_deselection(self, deselected, dataset_meta_features, learning_curves,
                                          algorithms_meta_features):
        """
        wrapper around preprocess to allow user to algin validation and test datasets in terms of algorithm
        deselection.
        """
        self.preprocess(dataset_meta_features,
                        {d: {a: curve for a, curve in algos.items() if int(a) not in deselected}
                         for d, algos in learning_curves.items()},
                        {k: v for k, v in algorithms_meta_features.items() if int(k) not in deselected})

    def _preprocess_learning_curves(self, algorithms_meta_features, learning_curves):
        """Enrich the learning curve objects with properties computed on the learning curve
        such as what the final performance is or when it reached the 90% convergence. etc."""
        # compute algorithm properties -----------------------------------------
        # compute learning curve properties.

        # find the 90% quantile in performance
        # ---> learn how fast an algo given a dataset complexity will approx converge?

        self.algo_learning_curves = {
            k: {} for k in algorithms_meta_features.keys()
        }
        for ds_id in learning_curves.keys():
            for algo_id, curve in learning_curves[ds_id].items():
                self.algo_learning_curves[str(algo_id)][ds_id] = curve

                # final performance
                curve.final_performance = curve.scores[-1]

                # how much in % of the final performance
                curve.convergence_share = curve.scores / curve.final_performance

                # stamp at which at least 90% of final performance is reached
                threshold = 0.9
                curve.convergence90_step = np.argmax(
                    curve.convergence_share >= threshold
                )
                curve.convergence90_time = curve.timestamps[curve.convergence90_step]
                curve.convergence90_performance = curve.scores[curve.convergence90_step]

    def _preporcess_scalar_properties(self, algo_curves):
        """
        calcualte scalar values for each algorithm, dataset combination

        :attributes: must be a single matrix [datasets, algorithms]
            -algo_convergences90_time: the time at which the algo reached its 90% performance
            on the respective dataset.
            -algo_90_performances: the algorithms observed performance, when
            it first surpassed the 90% threshold.
            -algo_final_performances: the final performance of algo on dataset

        """
        # read out the time of convergence 90%
        algo_convergences90_time = dict()
        algo_final_performances = dict()
        algo_90_performance = dict()
        for algo_id, curve_dict in algo_curves.items():
            algo_convergences90_time[algo_id] = pd.Series(
                {k: curve.convergence90_time for k, curve in curve_dict.items()})
            algo_final_performances[algo_id] = pd.Series(
                {k: curve.final_performance for k, curve in curve_dict.items()})
            algo_90_performance[algo_id] = pd.Series(
                {k: curve.convergence90_performance for k, curve in curve_dict.items()})

        # time at which the algo converged on the respective dataset
        self.algo_convergences90_time = pd.DataFrame(algo_convergences90_time)
        self.algo_convergences90_time.columns.name = 'Algorithms'
        self.algo_convergences90_time.index.name = 'Dataset'

        # how well did the algo perform on the respective dataset
        self.algo_final_performances = pd.DataFrame(algo_final_performances)
        self.algo_final_performances.columns.name = 'Algorithms'
        self.algo_final_performances.index.name = 'Dataset'

        # how was the performance at the timestamp that marks the at least 90% convergence
        self.algo_90_performances = pd.DataFrame(algo_90_performance)
        self.algo_90_performances.columns.name = 'Algorithms'
        self.algo_90_performances.index.name = 'Dataset'

    def _preprocess_meta_features(self, dataset_meta_features):
        """create a df with meta features of each dataset"""

        # Preprocess dataset meta data (remove the indiscriminative string variables)
        df = pd.DataFrame(
            list(dataset_meta_features.values()),
            index=dataset_meta_features.keys())

        binary_df = df[self.columns_binary].astype(float)

        # min-max normalization of numeric features
        numerical_df = df[self.columns_numerical].astype(float)
        numerical_df = self.enc_num.fit_transform(numerical_df)
        numerical_df = pd.DataFrame(
            numerical_df,
            columns=self.columns_numerical,
            index=dataset_meta_features.keys())

        categorical_df = df[self.columns_categorical]
        categorical_df = self.enc_cat.transform(categorical_df)
        categorical_df = pd.DataFrame(categorical_df, columns=self.enc_cat.get_feature_names(),
                                      index=dataset_meta_features.keys())

        self.datasets_meta_features_df = pd.concat([numerical_df, categorical_df, binary_df], axis=1)
        self.n_features = len(self.datasets_meta_features_df.columns)
        self.datasets_meta_features = torch.tensor(
            self.datasets_meta_features_df.values, dtype=torch.float32)

    def _preprocess_dataset_properties_meta_testing(self, dataset_meta_features):
        """create a dataset, that is of the exact same shape as the preprocessing"""
        # TODO check if dataset_meta_features is a single example : else: make it
        #  pd.Dataframe rather than a Series (and do the exact same as in preprocessing)
        df = pd.Series(dataset_meta_features).to_frame().T
        df = df[set(df.columns) - set(self.columns_descriptive)]

        binary_df = df[self.columns_binary].astype(float)

        # min-max normalization of numeric features
        numerical_df = df[self.columns_numerical].astype(float)
        numerical_df = self.enc_num.transform(numerical_df)
        numerical_df = pd.DataFrame(numerical_df, columns=self.columns_numerical)

        categorical_df = df[self.columns_categorical]
        categorical_df = self.enc_cat.transform(categorical_df)
        categorical_df = pd.DataFrame(categorical_df, columns=self.enc_cat.get_feature_names())

        self.datasets_meta_features_df = pd.concat([numerical_df, categorical_df, binary_df], axis=1)
        self.n_features = len(self.datasets_meta_features_df.columns)
        self.datasets_meta_features = torch.tensor(
            self.datasets_meta_features_df.values, dtype=torch.float32)

        return self.datasets_meta_features_df, self.datasets_meta_features

    def _preprocess_dataset_properties(self, learning_curves, dataset_meta_features):
        """
        Computes an object for each dataset and aggregates information for it.
        :param learning_curves:
        :return:
        """

        class Datum:
            # placeholder object to keep track of aggregation properties (instead of
            # all of the dictionaries flying around.
            pass

        self.dataset_learning_properties = {
            d: Datum() for d in learning_curves.keys()
        }
        b = 3  # number of points to konsider in the topk_available trials
        for d in learning_curves.keys():
            datum = self.dataset_learning_properties[d]
            datum.final_scores = []  # unconditional distribution
            datum.convergence_step = []
            datum.ranking = []
            datum.meta = dataset_meta_features[d]

            for a, curve in learning_curves[d].items():
                datum.convergence_step.append(curve.convergence90_step)
                # datum.convergence_avg_topk =
                datum.final_scores.append(curve.final_performance)

                datum.ranking.append(
                    (
                        a,
                        curve.final_performance,
                        curve.convergence90_step,
                        curve.convergence90_time,
                    )
                )

            datum.ranking = sorted(datum.ranking, key=lambda x: -x[1])

            datum.convergence_avg_topb = np.mean(
                [algotup[3] for algotup in datum.ranking[:b]]
            )
            datum.convergence_avg = np.mean(datum.convergence_step)

            datum.topk_available_trials = (
                    float(datum.meta["time_budget"]) / datum.convergence_avg_topb
            )

    def _preprocess_thresholded_algo_performances(self, k):
        """

        :param k: topk (how many algorithm performances are used for comparison in cosine similarity)
        all others that are not in topk performing on a dataset will be set to 0
        :return:
        """
        # order = self.datasets_meta_features_df.index
        # algo_performances = pd.DataFrame(
        #     [self.dataset_learning_properties[i].final_scores for i in order],
        #     index=order,
        # )  # index = dataset_id, column = algo_id
        #
        # self.algo_final_performances.columns

        # optional thresholding: remove k least performing
        algo_performances = torch.tensor(self.algo_final_performances.values, dtype=torch.float32)
        _, ind = torch.topk(algo_performances, k=k, largest=False, dim=1)
        for i_row, row in zip(ind, algo_performances):
            row[i_row] = 0

        self.algo_thresholded_performances = algo_performances

    def _reduce_algo_space(self, k=10, removals=5, mode='skew'):
        """
        Using backward elimination, remove the least performing algorithms.
        This helps reduce the complexity of the estimation problem.

        Core idea is to iteratively remove columns from self.algo_final_performances
        by choosing the ones that reduce the overall row score the least.
        """

        def calc_ecdf(data, n_bins=100):
            count, bins_count = np.histogram(data, bins=n_bins)

            # finding the PDF of the histogram using count values
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            return bins_count[1:], cdf

        def calc_pdf_integral0(data, n_bins):
            """

            :param data:
            :param n_bins:
            :return: probability of below x=0
            """
            x, cdf = calc_ecdf(data, n_bins)
            return cdf[np.argmax(x >= 0)]

        df = self.algo_final_performances
        k_inv = len(df.columns) - k

        # init for selection
        deselected = set()
        remaining = set(df.columns)

        performances = {k: None for k in df.loc[:, remaining]}
        performance_baseline = df.max(axis=1) - \
                               df[np.argsort(df, axis=1) >= k_inv].mean(axis=1, )
        skew_baseline = performance_baseline.skew(axis=0)
        for r in range(removals):

            # compute which algo hurts the least to remove:
            for algo in remaining:
                temp_remaining = set(remaining) - {algo}
                current_df = df.loc[:, temp_remaining]

                # METRIC for DESELECTION
                # compute average performance increase (on topk algos) on a dataset (value) deselecting this algo (key)
                # An improvement can come from deselecting a below mean performing algo
                # A deterioration comes from removing above mean performing algo on that dataset
                consider = k_inv - len(deselected)  # rank threshold (minimum rank to get into topk)
                absolute_perf = current_df[np.argsort(current_df, axis=1) >= consider].mean(axis=1)
                performances[algo] = performance_baseline - (current_df.max(axis=1) - absolute_perf)

            if mode == 'skew':
                # remove the algo, that reduces the skewness of the baseline the most
                skew = pd.DataFrame(performances).skew(axis=0)
                deselected_algo = skew.index[np.argmin(skew_baseline - skew)]
                skew_baseline = skew[deselected_algo]

            elif mode == '0threshold':
                # remove the algo that has least density mass on reducing the overall
                # performance of datasets (compared to current baseline)
                deselected_algo = pd.Series({k: calc_pdf_integral0(v, n_bins=100)
                                             for k, v in performances.items()}).argmin()

            deselected.add(deselected_algo)
            remaining.remove(deselected_algo)
            performance_baseline = performances.pop(deselected_algo)

            # # plotting the change in baseline performances across datasets:
            # # this is the performance profile across remaining algos & datasets removing an algo.
            # for algo in performances.keys():
            #     # performances[algo].hist()
            #
            #     performances[algo].hist(density=True, histtype='step',
            #                             cumulative=True, label=str(algo), bins=100, )
            # plt.legend()
            # plt.title(f'Leave this algo out decrease in top-{k} avg. performance on a dataset')
            # plt.show()

        return deselected

    def plot_learning_curves(self, dataset_id=9):

        for id, curve in self.raw_learning_curves[str(dataset_id)].items():
            plt.plot(curve.timestamps, curve.scores, marker='o', linestyle='dashed', label=id)

        plt.title('Dataset {}'.format(dataset_id))
        plt.xlabel('Time')
        plt.ylabel('Performance')
        plt.legend(loc='upper right')
        plt.show()

    def plot_convergence90_time(self, normalized=True):

        for dataset_id, dataset_curves in self.raw_learning_curves.items():

            if normalized:
                # fixme: apparently time budget can be smaller than timestamp
                # unconditional_90time = [
                #     curve.convergence90_time / int(self.raw_dataset_meta_features[str(dataset_id)]['time_budget'])
                #     for curve in dataset_curves.values()]

                unconditional_90time = [
                    curve.convergence90_time / curve.timestamps[-1]
                    for curve in dataset_curves.values()]
                title = 'Unconditional Time Budget until 90% Convergence\n' \
                        'normalized by available budget'
            else:
                unconditional_90time = [curve.convergence90_time for curve in dataset_curves.values()]
                title = 'Unconditional Time Budget until 90% Convergence'
            sns.kdeplot(unconditional_90time, label=dataset_id)

        plt.title(title)
        plt.legend(loc='upper right')
        plt.show()
