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

def get_cc18_datasets():
    cc18_benchmark_suite = openml.study.get_suite(99)
    return openml.datasets.get_datasets(cc18_benchmark_suite.data)


def evaluate_algorithm_on_dataset(openml_dataset, algorithm, budgets, number_of_splits, seed):
    print(f"{openml_dataset}")
    pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                               ('onehot', OneHotEncoder(handle_unknown='ignore')),
                               ('classifier', algorithm)])

    X, y, categorical_indicator, attribute_names = openml_dataset.get_data(
        dataset_format="dataframe",
        target=openml_dataset.default_target_attribute)
    X = X.to_numpy()
    y = y.to_numpy()

    stratiefiedKFold = StratifiedKFold(n_splits=number_of_splits, random_state=seed, shuffle=True)
    stratified_X_trains = list()
    stratified_y_trains = list()
    stratified_X_tests = list()
    stratified_y_tests = list()

    for train_index, test_index in stratiefiedKFold.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        stratified_X_trains.append(X_train)
        stratified_X_tests.append(X_test)

        y_train, y_test = y[train_index], y[test_index]
        stratified_y_trains.append(y_train)
        stratified_y_tests.append(y_test)

    for budget in budgets:
        accuracies = list()

        for fold in range(number_of_splits):
            # sample down to the correct size
            X_subsampled, y_subsampled = resample(
                stratified_X_trains[fold],
                stratified_y_trains[fold],
                random_state=fold,
                n_samples=int(budget*stratified_X_trains[fold].shape[0]))

            # train pipeline
            pipeline.fit(X_subsampled, y_subsampled)

            # predict and compute accuracy
            predictions = pipeline.predict(stratified_X_tests[fold])
            accuracy = accuracy_score(stratified_y_tests[fold], predictions)
            accuracies.append(accuracy)

        average_accuracy_over_folds = np.asarray(accuracies).mean()
        print(f"budget: {budget}, accuracy: {average_accuracy_over_folds}")

        # TODO push score to database


if __name__ == "__main__":
    datasets = get_cc18_datasets()
    budgets = [0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    number_of_splits = 3

    seed = 42
    np.random.seed(seed)

    for dataset in tqdm(datasets):
        evaluate_algorithm_on_dataset(dataset, RandomForestClassifier(), budgets, number_of_splits, seed)