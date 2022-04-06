import os
from sys import argv, path
root_dir = os.path.abspath(os.curdir)
path.append(root_dir)
import json
import numpy as np
import math
import csv
from learning_curve import Learning_Curve

#=== Verbose mode
verbose = True

def vprint(mode, t):
    """
    Print to stdout, only if in verbose mode.

    Parameters
    ----------
    mode : bool
        True if the verbose mode is on, False otherwise.

    Examples
    --------
    >>> vprint(True, "hello world")
    hello world

    >>> vprint(False, "hello world")

    """

    if(mode):
        print(str(t))

class Meta_Learning_Environment():
    """
    A meta-learning environment which provides access to all meta-data.
    """

    def __init__(self, validation_data_dir, test_data_dir, meta_features_dir, algorithms_meta_features_dir, output_dir):
        """
        Initialize the meta-learning environment

        Parameters
        ----------
        validation_data_dir : str
            Path to learning curve data on the validation set
        test_data_dir : str
            Path to learning curve data on the test set
        meta_features_dir : str
            Path to meta features of datasets
        algorithms_meta_features_dir : str
            Path to algorithms_meta_features of algorithms
        output_dir : str
            Path to output directory

        """
        self.output_dir = output_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.meta_features_dir = meta_features_dir
        self.algorithms_meta_features_dir = algorithms_meta_features_dir
        self.num_dataset = 1

        #=== List of dataset names
        self.list_datasets = os.listdir(self.test_data_dir)
        if '.DS_Store' in self.list_datasets:
            self.list_datasets.remove('.DS_Store')
        self.list_datasets.sort()

        #=== List of algorithms
        self.list_algorithms = os.listdir(os.path.join(self.test_data_dir, self.list_datasets[0]))
        if '.DS_Store' in self.list_algorithms:
            self.list_algorithms.remove('.DS_Store')
        self.list_algorithms.sort()
        print("self.list_algorithms.sort() = ", self.list_algorithms)
        self.num_algo = len(self.list_algorithms)

        self.load_all_data()

    def load_all_data(self):
        """
        Load all data
        """
        self.validation_learning_curves = {}
        self.test_learning_curves = {}
        self.meta_features = {}
        self.algorithms_meta_features = {}

        #=== Load META-FEATURES
        vprint(verbose, "[+]Start loading META-FEATURES of datasets")
        # Iterate through all datasets
        for d in os.listdir(self.meta_features_dir):
            if '.DS_Store' not in d:
                dataset_name = d.split('.')[0].split('_')[0]
                dict_temp = {}
                with open(os.path.join(self.meta_features_dir, d), 'r') as f:
                    for line in f:
                        key, value = line.split('=')
                        key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                        dict_temp[key] = value
                self.meta_features[dataset_name] = dict_temp
        vprint(verbose, "[+]Finished loading META-FEATURES of datasets")

        #=== Load HYPERPARAMETERS
        vprint(verbose, "[+]Start loading HYPERPARAMETERS of algorithms")
        # Iterate through all datasets
        for d in os.listdir(self.algorithms_meta_features_dir):
            if '.DS_Store' not in d:
                algorithm_name = d.split('.')[0]
                dict_temp = {}
                with open(os.path.join(self.algorithms_meta_features_dir, d), 'r') as f:
                    for line in f:
                        key, value = line.split('=')
                        key, value = key.replace(' ','').replace('\n', ''), value.replace(' ','').replace('\n', '').replace('\'','') #remove whitespaces and special symbols
                        dict_temp[key] = value
                self.algorithms_meta_features[algorithm_name] = dict_temp
        vprint(verbose, "[+]Finished loading HYPERPARAMETERS of algorithms")

        #=== Load VALIDATION LEARNING CURVES
        if self.validation_data_dir!=None:
            vprint(verbose, "[+]Start loading VALIDATION learning curves")
            # Iterate through all datasets
            for dataset_name in self.list_datasets:
                dict_temp = {}
                for algo_name in self.list_algorithms:
                    path_to_algo = os.path.join(self.validation_data_dir, dataset_name, algo_name)
                    dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"))
                self.validation_learning_curves[dataset_name] = dict_temp
            vprint(verbose, "[+]Finished loading VALIDATION learning curves")

        #=== Load TEST LEARNING CURVES
        vprint(verbose, "[+]Start loading TEST learning curves")
        # Iterate through all datasets
        for dataset_name in self.list_datasets:
            dict_temp = {}
            for algo_name in self.list_algorithms:
                path_to_algo = os.path.join(self.test_data_dir, dataset_name, algo_name)
                dict_temp[algo_name] = Learning_Curve(os.path.join(path_to_algo + "/scores.txt"))
            self.test_learning_curves[dataset_name] = dict_temp
        vprint(verbose, "[+]Finished loading TEST learning curves")

    def reset(self, dataset_name):
        """
        Reset the environment for a new task

        Parameters
        ----------
        dataset_name : str
            Name of the dataset at hand

        Returns
        ----------

        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features, algorithms_meta_features = env.reset("waldo")
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}
        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """
        self.dataset_name = dataset_name
        self.counters = {i:0.0 for i in self.list_algorithms} # Counters keeping track of the time has been spent for each algorithm
        dataset_meta_features = self.meta_features[dataset_name]
        self.total_time_budget = float(dataset_meta_features['time_budget'])
        self.remaining_time_budget = self.total_time_budget

        return dataset_meta_features, self.algorithms_meta_features

    def reveal(self, action):
        """
        Execute the action and reveal new information on the learning curves

        Parameters
        ----------
        action : tuple of (int, int, float)
            The suggested action consisting of 3 things:
                (1) A_star: algorithm for revealing the next point on its test learning curve
                            (which will be used to compute the agent's learning curve)
                (2) A:  next algorithm for exploring and revealing the next point
                       on its validation learning curve
                (3) delta_t: time budget will be allocated for exploring the chosen algorithm in (2)
        Returns
        ----------
        observation : tuple of (int, float, float)
                An observation containing:
                    1) A: the explored algorithm,
                    2) C_A: time has been spent for A
                    3) R_validation_C_A: the validation score of A given C_A
        done : bool
            True if the time budget is exhausted, False otherwise

        Examples
        ----------
        >>> observation, done = env.reveal((9, 9, 80))
        >>> observation
            (9, 151.73, 0.5)
        >>> done
            True

        """
        A_star, A, delta_t = action
        done = False

        #=== Limit the delta_t from 0 to remaining_time_budget:
        delta_t = min(max(delta_t, 0), self.remaining_time_budget)

        #=== Check done
        if delta_t >= self.remaining_time_budget:
            done = True

        #=== Reveal C_A_star, R_test_C_A_star
        if A_star==None:
            C_A_star = 0.0
        else:
            _, C_A_star = self.test_learning_curves[self.dataset_name][str(A_star)].get_last_point_within_delta_t(0, self.counters[str(A_star)])

        #=== Reveal C_A, R_validation_C_A
        R_validation_C_A, C_A = self.validation_learning_curves[self.dataset_name][str(A)].get_last_point_within_delta_t(delta_t, self.counters[str(A)])

        #=== Update algorithm counters and the remaining time budget
        self.counters[str(A)] = C_A
        self.remaining_time_budget = self.remaining_time_budget - delta_t

        #=== Prepare an observation to be sent to the agent
        observation = (A, C_A, R_validation_C_A)

        #=== Write hidden_observation to the ouput directory
        # if A_star!=None:
        total_time_spent = np.around(self.total_time_budget-self.remaining_time_budget, decimals=2)
        hidden_observation = [str(A_star), str(C_A_star), total_time_spent]
        with open(self.output_dir + '/' + self.dataset_name + '.csv', 'a') as f:
            writer = csv.writer(f) # create the csv writer
            writer.writerow(hidden_observation) # write a row to the csv file

        return observation, done
