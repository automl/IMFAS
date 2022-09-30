import logging
from pathlib import Path

import torch
from lcdb import get_curve, get_meta_features, get_all_curves
from tqdm import tqdm

from imfas.data.eda.lcdb import dataset_anchors, fidelities217, fidelities216, fidelities175, \
    fidelities126, fidelities95, fidelities61, fidelities41, fidelities31

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class LCDB:
    available_dataset_openml_ids = None

    # available _fidelities for a dataset. set of tuples, to check during collection
    # if the assumption is true, that all these _fidelities are available for all algos in the
    # dataset.
    dataset_anchors = {k: {tuple(sorted(v))} for k, v in dataset_anchors.items()}

    # catching all errors during loading of the curves
    loading_errors = {}

    available_fidelities = [  # Just the default version with most datasets available
        fidelities217,
        fidelities216,
        fidelities175,
        fidelities126,
        fidelities95,
        fidelities61,
        fidelities41,
        fidelities31
    ]

    n_spits = 25
    n_seeds = 1

    def __init__(self, path: Path, fidelities: set, metric):
        """
        Choosing the set of _fidelities one would want to have implies the amount
        of available datasets.

        :path: path where to the datasets are to be stored.
        :param fidelities: set of fidelity labels to be used
        """
        self._fidelities = fidelities
        self.path = path
        self.metric = metric

        # conditioned on the _fidelities, which datasets are available?
        self.dataset_ids = self._find_datasets_with_fidelities(fidelities)
        self.algo_names = self._get_algo_names()

        # create folders
        path_train = self.path / "lcs_train_datasets"
        path_valid = self.path / "lcs_valid_datasets"
        path_test = self.path / "lcs_test_datasets"

        for path in [path_train, path_valid, path_test]:
            path.mkdir(parents=True, exist_ok=True)

        # dataset_id (openml), algo_name
        self.short = {}  # not entire tensor is used (only the first k lc are availale
        self.skipped = []  # skipped because of errors
        self.no_curves = []  # skipped because no curves were found
        self.filled = {}  # dataset_id, list[indicies of algos]

    @property
    def n_datasets(self):
        return len(self.dataset_ids)

    @property
    def n_algos(self):
        return len(self.algo_names)

    @property
    def shape(self):
        return self.n_datasets, self.n_algos, self.n_spits * self.n_seeds, len(self._fidelities)

    @property
    def fidelities(self):
        return list(sorted(self._fidelities))

    def __repr__(self):
        return f"LCDB(_fidelities={self._fidelities}," \
               f" n_datasets={self.n_datasets}, n_algos={self.n_algos})"

    def _find_datasets_with_fidelities(self, fidelities: set, ):
        D_fidelity = {d: set(*v) for d, v in self.dataset_anchors.items()}
        datasets = [d for d, v in D_fidelity.items() if fidelities.issubset(v)]

        log.info(f"Found {len(datasets)} datasets that may exhibit _fidelities {fidelities}")

        return datasets

    def _get_algo_names(self):
        curves = get_all_curves(self.metric)
        algo_names = set(curves['learner'])
        del curves

        return algo_names

    def get_curves(self, dataset_id, algo_name, metric):
        """
        Curves of a dataset-algorithm combination.

        Felix mohr collected 25 fold repeitions with 5 different seeds for each
        dataset-algorithm combination.

        :return tuple[tensor, tensor, tensor], training, validation, testing curves.
        Notably, each of these tensors is of size (n_fidelities, seeds*folds)
        """
        try:
            subset_sizes, scores_train, scores_valid, scores_test = get_curve(
                dataset_id,  # openmlid
                algo_name,  # learner
                metric  # metric
            )

            # fixme: needs to subset the _fidelities here!

            self.dataset_anchors[dataset_id].add(tuple(subset_sizes))

            # assert len(self.dataset_anchors[dataset_id]) == 1, \
            #     "The assumption that all algorithms have the same _fidelities is violated."
            # # TODO: how to deal with violations?
            #
            # assert len(subset_sizes) == len(self._fidelities), \
            #     "The assumption that all algorithms have the same _fidelities is violated." \
            #     f"The _fidelities of the dataset are: {self._fidelities},\n" \
            #     f" but the _fidelities of the algorithm are: {subset_sizes}"

            assert len(subset_sizes) == len(scores_train), \
                f"fidelities and tensorshapes do not match for {dataset_id}, {algo_name}"
            assert len(subset_sizes) == len(scores_valid), \
                f"fidelities and tensorshapes do not match for {dataset_id}, {algo_name}"
            assert len(subset_sizes) == len(scores_test), \
                f"fidelities and tensorshapes do not match for {dataset_id}, {algo_name}"

            tr = torch.tensor(scores_train).T
            va = torch.tensor(scores_valid).T
            te = torch.tensor(scores_test).T

            # select the relevant fidelity set:
            fid_idx = [subset_sizes.index(f) for f in self.fidelities]

            tr = tr[:, fid_idx]
            va = va[:, fid_idx]
            te = te[:, fid_idx]

            assert all([tr.shape == va.shape, va.shape == te.shape]), \
                f"Fold*seed dimensions do not match across train, valid, test for {dataset_id}, {algo_name}"

            return tr, va, te

        except Exception as e:
            if "No curves found for" in str(e):
                self.no_curves.append((dataset_id, algo_name))
            else:
                self.skipped.append((dataset_id, algo_name))
                self.loading_errors[(dataset_id, algo_name)] = e

            log.debug(f"skipping ({dataset_id}, {algo_name}) because of \n{e}")

            return None, None, None

    def load_dataset_meta_features(self):
        """
        Of the relevant subset (conditioned on the _fidelities chosen) of datasets.
        """
        dataset_meta_features = get_meta_features()
        dataset_meta_features.index = dataset_meta_features['openmlid']
        LCDB.available_dataset_openml_ids = dataset_meta_features.openmlid.unique()

        # select the relevant datasets
        dataset_meta_features = dataset_meta_features.loc[self.dataset_ids]
        dataset_meta_features.index = dataset_meta_features['Name']
        return dataset_meta_features

    def collect_all_data(self):
        """
        Given the _fidelities, we need to subset the dataset meta features & curves to
        the relevant datasets.
        Their Learning curves need to be stored to disk.
        """

        dataset_meta_features = self.load_dataset_meta_features()
        dataset_meta_features.to_csv(self.path / 'dataset_meta_features.csv')

        shape = (len(self.algo_names), self.n_spits * self.n_seeds, len(self._fidelities))
        for d, dataset_id in tqdm(enumerate(self.dataset_ids), desc="Datasets"):
            train = torch.zeros(shape) * torch.nan
            valid = torch.zeros(shape) * torch.nan
            test = torch.zeros(shape) * torch.nan

            for a, algo_name in enumerate(self.algo_names):
                tr, va, te = self.get_curves(dataset_id, algo_name, self.metric)

                if tr is None:
                    continue

                try:
                    if tr.shape[0] == 125:
                        # eg. dataset openmlid==6
                        tr = tr.reshape(5, 25, -1)[0]
                        va = va.reshape(5, 25, -1)[0]
                        te = te.reshape(5, 25, -1)[0]
                    if tr.shape[0] < 25:
                        s = tr.shape[0]
                        train[a, s:, :] = tr
                        valid[a, s:, :] = va
                        test[a, s:, :] = te
                        self.short[(d, a)] = s
                    else:
                        train[a, :, :] = tr
                        valid[a, :, :] = va
                        test[a, :, :] = te

                except Exception as e:
                    self.skipped.append((dataset_id, algo_name))
                    log.debug(f"skipping ({dataset_id}, {algo_name}) because of \n{e}")
                    self.loading_errors[(dataset_id, algo_name)] = e

            # log.info(f"TrainTensor: {train.shape}, of memory footprint: {sys.getsizeof(train)}")
            filled = [i for i, t in enumerate(train) if not torch.equal(t, torch.zeros_like(t))]
            self.filled[d] = filled
            log.debug(f"{filled} algorithms are (somewhat) available for {dataset_id}")

            torch.save(train, self.path / f"{str(d)}.pt")
            torch.save(valid, self.path / f"{str(d)}.pt")
            torch.save(test, self.path / f"{str(d)}.pt")

        log.debug(f"Saving {len(self.loading_errors)} errors to {self.path / 'loading_errors.txt'}")

        log.debug(f"available fidelities in the dataset (observed for all algorithms):"
                  f" {self.dataset_anchors}")

        log.debug(f'--------------------------SUMMARY--------------------------'
                  f'skipped openml_ids {self.skipped}\n\n'
                  f'tensor are filled at indicies {self.filled}\n\n'
                  f'at those indicies some are short, and only filled up till: {self.short}\n\n'
                  f'couldn\'t query these openmlids {self.no_curves}')

        with open(self.path / 'loading_errors.txt', 'w') as f:
            f.write(str(self.loading_errors))


if __name__ == '__main__':
    fidelities = set.union(LCDB.fidelities[:-1])
