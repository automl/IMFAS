from omegaconf import DictConfig

from typing import Callable, Dict, Optional, Union

from imfas.trainer import BaseTrainer

from hydra.utils import instantiate
import torch
import wandb

from tqdm import tqdm


# srun python main.py +experiment=${MODEL}  dataset=${DATASET} wandb.mode=online seed=$SLURM_ARRAY_TASK_ID train_test_split.fold_idx=${FOLD_IDX} # dataset.dataset_raw.enable=true
#SBATCH --array=1-5
# for FOLDIDX in {0..9}
# default config

# TODO debug with new scaling of fidelities
# TODO check the device is GPU


class LCNetTrainer(BaseTrainer):

    def __init__(
            self,
            model,

            batch_size,
            keep_every,
            verbose=False,
            lr: float = 1e-2,
            epsilon: float = 1e-10,
            mdecay: float = 0.05,
            continue_training: bool = False,
            optimizer=None,
            sampler='adaptive_sghmc'
    ):
        """
        FIXME: This must be the most ugly class in the entire project
        :param verbose: verbose output
        :param batch_size: batch size

        :param num_steps: Number of sampling steps to perform after burn-in is finished.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param keep_every: Number of sampling steps (after burn-in) to perform before keeping a sample.
            In total, `num_steps // keep_every` network weights will be sampled."""

        self.model_wrapper = model
        self.optimizer = None  # fixme: this is a hack to meet the interface

        self.batch_size = batch_size

        self.keep_every = keep_every
        self.verbose = verbose

        self.lr = lr
        self.epsilon = epsilon
        self.mdecay = mdecay
        self.continue_training = continue_training
        self.sampling_method = sampler

        self.do_normalize_input = True
        self.do_normalize_output = False
        self.print_every_n_steps = 100

    def _parse_single_dataset(self, dataset, dataset_id):
        learning_curves, dataset_meta_features, algo_meta_features = \
            dataset.learning_curves, dataset.meta_dataset, dataset.meta_algo

        self.n_fidelity = learning_curves.shape[-1]


        self.fidelities =  torch.linspace(0.001, 1., self.n_fidelity)
        n_algos = algo_meta_features.shape[0]

        # filtering out constant columns (which throw an error in normalization)
        keep_cols = torch.std(algo_meta_features.transformed_df, 0) != 0

        # prepare x_train
        fid = self.fidelities.repeat(n_algos).view(-1, 1)
        x_train = torch.repeat_interleave(
            algo_meta_features.transformed_df[:, keep_cols], 51, dim=0
        )
        x_train = torch.cat((x_train, fid), dim=1)

        # fixme: we need to train on a fixed dataset
        # prepare y_train
        y_train = learning_curves.transformed_df[dataset_id]

        # final fidelity as prompt & target
        x_test = torch.cat((algo_meta_features.transformed_df[:, keep_cols],
                            torch.ones((n_algos, 1))),
                           dim=1)
        y_test = learning_curves.transformed_df[dataset_id, :, -1]

        return x_train, y_train, x_test, y_test

    def run(
            self,
            train_loader: torch.utils.data.DataLoader = None,
            valid_loader: torch.utils.data.DataLoader = None,
            test_loader: torch.utils.data.DataLoader = None,
            epochs: int = 0,
            log_freq: int = 5,
            train_loss_fn: Union[Callable, DictConfig] = None,
            valid_loss_fns: Dict[str, Callable] = None,
            test_loss_fns: Dict[str, Callable] = None,
            num_steps: int = 1000,
            num_burn_in_steps: int = 100,
            **train_kwargs,
    ):

        # just to have the self.n_fidelity attribute :(
        _, _, _, _ = self._parse_single_dataset(test_loader.dataset, 0)

        # average over all test datasets
        for loss_fn in test_loss_fns.keys():
            wandb.log({f"Test, Slice Evaluation: {loss_fn}": float("nan"),
                       "fidelity": 0})

            continue
        for f in tqdm(range(1, self.n_fidelity)):
            losses = {k: torch.zeros(len(test_loader)) for k in test_loss_fns.keys()}

            for i, d in enumerate(test_loader.dataset.split):

                x_train, y_train, x_test, y_test = self._parse_single_dataset(
                    train_loader.dataset, d)

                # constrain x_train & y_train up to fidelity f
                ind = (self.fidelities <= self.fidelities[f-1]).repeat(self.n_fidelity - 1)
                x_train_f = x_train[ind, :]
                y_train_f = y_train[:, :f ]

                self.model_wrapper.train(
                    x_train_f.detach().numpy(),
                    y_train_f.flatten().detach().numpy(),
                    num_steps=num_steps,
                    num_burn_in_steps=num_burn_in_steps,
                    batch_size=self.batch_size,
                    keep_every=self.keep_every,
                    verbose=self.verbose,
                    lr=self.lr,
                    epsilon=self.epsilon,
                    mdecay=self.mdecay,
                    continue_training=self.continue_training,
                    sampling_method=self.sampling_method,
                )

                x_train, y_train, x_test, y_test = self._parse_single_dataset(
                    test_loader.dataset, d
                )
                for loss_name, loss_fn in test_loss_fns.items():
                    m, v = self.model_wrapper.predict(x_test.detach().numpy())
                    losses[loss_name][i] = instantiate(loss_fn)(torch.tensor(m).view(1, -1),
                                                                y_test.view(1, -1))

            # average over all test datasets
            for loss_fn in test_loss_fns.keys():
                wandb.log({f"Test, Slice Evaluation: {loss_fn}": losses[loss_fn].mean(),
                           "fidelity": f})

        # just to meet the interface. fixme: this is a hack
        return torch.ones(1), torch.ones(1)
