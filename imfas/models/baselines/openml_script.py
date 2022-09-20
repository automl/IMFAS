import openml
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import multiprocessing
from joblib import Parallel, delayed

import pandas as pd

import pdb


def get_cc18_datasets(n_datasets):
    cc18_benchmark_suite = openml.study.get_suite(1)

    return openml.datasets.get_datasets(cc18_benchmark_suite.data)


def evaluate_algorithm_on_dataset(openml_dataset, algorithm, budgets, number_of_splits, seed):
    print(f"{openml_dataset}")

    # NOTE Dataframes for CSV processing
    dataset_mf = pd.DataFrame()
    algorithm_mf = pd.DataFrame()
    learning_curves = pd.DataFrame()

    # Define a classifcation pipeline
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", algorithm),
        ]
    )

    # Get the features from the openml dataset

    X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
        dataset_format="dataframe", target=openml_dataset.default_target_attribute
    )
    X = X.to_numpy()
    y = y.to_numpy()

    # initialitze the stratified folds
    stratiefiedKFold = StratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    stratified_X_trains = list()
    stratified_y_trains = list()
    stratified_X_tests = list()
    stratified_y_tests = list()

    # Get the features and data for each fold
    for train_index, test_index in stratiefiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        stratified_X_trains.append(X_train)
        stratified_X_tests.append(X_test)

        y_train, y_test = y[train_index], y[test_index]
        stratified_y_trains.append(y_train)
        stratified_y_tests.append(y_test)

    # Run the algorithm for each budget for all folds
    for budget in budgets:
        accuracies = list()

        for fold in range(number_of_splits):
            # sample down to the correct size
            X_subsampled, y_subsampled = resample(
                stratified_X_trains[fold],
                stratified_y_trains[fold],
                random_state=fold,
                n_samples=int(budget * stratified_X_trains[fold].shape[0]),
            )

            # train pipeline
            pipeline.fit(X_subsampled, y_subsampled)

            # predict and compute accuracy
            predictions = pipeline.predict(stratified_X_tests[fold])
            accuracy = accuracy_score(stratified_y_tests[fold], predictions)
            accuracies.append(accuracy)

        average_accuracy_over_folds = np.asarray(accuracies).mean()
        print(f"budget: {budget}, accuracy: {average_accuracy_over_folds}")

    # TODO Log into CSV


if __name__ == "__main__":
    datasets = get_cc18_datasets(n_datasets=2)

    pdb.set_trace()

    budgets = np.arange(0.05, 1.05, 0.05, dtype=float)
    number_of_splits = 3

    seed = 42
    np.random.seed(seed)

    algorithms = [RandomForestClassifier()]

    # Function for parallell running
    def calc_stuff():

        for algo in algorithms:
            for dataset in datasets:
                evaluate_algorithm_on_dataset(dataset, algo, budgets, number_of_splits, seed)

    # TODO parallelize based dataset computation time

    Parallel(n_jobs=2)(delayed(calc_stuff))  # NOTE yet to be done
