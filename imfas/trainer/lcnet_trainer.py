import time
from itertools import islice

from omegaconf import DictConfig

import logging

from typing import Callable, Dict, Optional, Union

import numpy as np
import torch

import torch.utils.data as data_utils

from pybnn.priors import weight_prior, log_variance_prior
from pybnn.sampler import   SGLD, SGHMC, PreconditionedSGLD #, AdaptiveSGHMC
from pybnn.util.infinite_dataloader import infinite_dataloader
from pybnn.util.normalization import zero_mean_unit_var_denormalization
from scipy.stats import norm

from copy import deepcopy

from hydra.utils import instantiate

from imfas.trainer import BaseTrainer

import torch

from torch.optim import Optimizer


class AdaptiveSGHMC(Optimizer):
    """ Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses a burn-in
        procedure to adapt its own hyperparameters during the initial stages
        of sampling.
        See [1] for more details on this burn-in procedure.\n
        See [2] for more details on Stochastic Gradient Hamiltonian Monte-Carlo.
        [1] J. T. Springenberg, A. Klein, S. Falkner, F. Hutter
            In Advances in Neural Information Processing Systems 29 (2016).\n
            `Bayesian Optimization with Robust Bayesian Neural Networks. <http://aad.informatik.uni-freiburg.de/papers/16-NIPS-BOHamiANN.pdf>`_
        [2] T. Chen, E. B. Fox, C. Guestrin
            In Proceedings of Machine Learning Research 32 (2014).\n
            `Stochastic Gradient Hamiltonian Monte Carlo <https://arxiv.org/pdf/1402.4102.pdf>`_
    """

    def __init__(self,
                 params,
                 lr: float = 1e-2,
                 num_burn_in_steps: int = 3000,
                 epsilon: float = 1e-16,
                 mdecay: float = 0.05,
                 scale_grad: float = 1.) -> None:
        """ Set up a SGHMC Optimizer.
        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr: float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        num_burn_in_steps: int, optional
            Number of burn-in steps to perform. In each burn-in step, this
            sampler will adapt its own internal parameters to decrease its error.
            Set to `0` to turn scale adaption off.
            Default: `3000`.
        epsilon: float, optional
            (Constant) per-parameter epsilon level.
            Default: `0.`.
        mdecay:float, optional
            (Constant) momentum decay per time-step.
            Default: `0.05`.
        scale_grad: float, optional
            Value that is used to scale the magnitude of the epsilon used
            during sampling. In a typical batches-of-data setting this usually
            corresponds to the number of examples in the entire dataset.
            Default: `1.0`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError("Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, scale_grad=float(scale_grad),
            num_burn_in_steps=num_burn_in_steps,
            mdecay=mdecay,
            epsilon=epsilon
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]

                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(parameter)
                    state["g"] = torch.ones_like(parameter)
                    state["v_hat"] = torch.ones_like(parameter)
                    state["momentum"] = torch.zeros_like(parameter)
                state["iteration"] += 1

                mdecay, epsilon, lr = group["mdecay"], group["epsilon"], group["lr"]
                scale_grad = torch.tensor(group["scale_grad"], dtype=parameter.dtype)
                tau, g, v_hat = state["tau"], state["g"], state["v_hat"]

                momentum = state["momentum"]
                gradient = parameter.grad.data * scale_grad

                tau_inv = 1. / (tau + 1.)

                # update parameters during burn-in
                if state["iteration"] <= group["num_burn_in_steps"]:
                    tau.add_(- tau * (
                            g * g / (v_hat + epsilon)) + 1)  # specifies the moving average window, see Eq 9 in [1] left
                    g.add_(-g * tau_inv + tau_inv * gradient)  # average gradient see Eq 9 in [1] right
                    v_hat.add_(-v_hat * tau_inv + tau_inv * (gradient ** 2))  # gradient variance see Eq 8 in [1]

                minv_t = 1. / (torch.sqrt(v_hat) + epsilon)  # preconditioner

                epsilon_var = (2. * (lr ** 2) * mdecay * minv_t - (lr ** 4))

                # sample random epsilon
                sigma = torch.sqrt(torch.clamp(epsilon_var, min=1e-16))
                sample_t = torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient) * sigma)

                # update momentum (Eq 10 right in [1])
                momentum.add_(
                    - (lr ** 2) * minv_t * gradient - mdecay * momentum + sample_t
                )

                # update theta (Eq 10 left in [1])
                parameter.data.add_(momentum)

        return loss


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

    @staticmethod
    def normalize_input(x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)

        x_norm = deepcopy(x)

        print(f'std: {s[:-1]}, mean: {m[:-1]}, x: {x[:, :-1]}')

        # check if std is 0
        if np.any(s[:-1] == 0):
            raise ValueError("std is 0 in lcnet.normalize_input")

        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s

    def train(self, train_loader, epochs, loss_fn, log_freq=1, num_steps=13000,
              num_burn_in_steps=3000):
        """
        :param train_loader: torch.utils.data.DataLoader (but actually the one we used earlier
        for all other methods) -- this is a hack to meet the interface (with little additions)
        MOST UGGLY CODE
        """
        x_train, y_train, x_test, y_test = self._parse_single_dataset(
            train_loader.dataset, dataset_id=0
        )  # fixme: adjust this to allow multiple datasets

        train_loader, x_train_, y_train_ = self._instantiate_train_loader(
            x_train, y_train, batch_size=self.batch_size
        )

        self._instantiate_sampler(self.num_datapoints, self.input_dimensionality,
                                  num_burn_in_steps,
                                  self.continue_training, self.lr, self.mdecay, self.epsilon)

        batch_generator = islice(enumerate(train_loader), num_steps)

        logging.debug("Training started.")
        start_time = time.time()
        for step, (x_batch, y_batch) in batch_generator:
            self.sampler.zero_grad()
            loss = loss_fn(input=self.model(x_batch), target=y_batch)
            loss.backward()
            self.sampler.step()

            # Add prior. Note the gradient is computed by: g_prior + N/n sum_i grad_theta_xi see Eq 4
            # in Welling and Whye The 2011. Because of that we divide here by N=num of datapoints since
            # in the sample we rescale the gradient by N again
            loss -= log_variance_prior(
                self.model(x_batch)[:, 1].view((-1, 1))) / self.num_datapoints
            loss -= weight_prior(self.model.parameters(), dtype=self.dtype) / self.num_datapoints

            # # print the requires grad status of the model parameters
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)

            loss.backward()
            self.sampler.step()

            if self.verbose and step > 0 and step % self.model.print_every_n_steps == 0:

                # compute the training performance of the ensemble
                if len(self.model.sampled_weights) > 1:
                    mu, var = self.model.predict(x_train)
                    total_nll = -np.mean(norm.logpdf(y_train, loc=mu, scale=np.sqrt(var)))
                    total_mse = np.mean((y_train - mu) ** 2)
                # in case we do not have an ensemble we compute the performance of the last weight sample
                else:
                    f = self.model(x_train_)

                    if self.do_normalize_output:
                        mu = zero_mean_unit_var_denormalization(f[:, 0], self.y_mean,
                                                                self.y_std).data.numpy()
                        var = torch.exp(f[:, 1]) * self.y_std ** 2
                        var = var.data.numpy()
                    else:
                        mu = f[:, 0].data.numpy()
                        var = np.exp(f[:, 1].data.numpy())
                    total_nll = -np.mean(norm.logpdf(y_train, loc=mu, scale=np.sqrt(var)))
                    total_mse = np.mean((y_train - mu) ** 2)

                t = time.time() - start_time

                if step < num_burn_in_steps:
                    print("Step {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                          "Time = {:5.2f}".format(step, float(total_nll),
                                                  float(total_mse), t))

                if step > num_burn_in_steps:
                    print("Step {:8d} : NLL = {:11.4e} MSE = {:.4e} "
                          "Samples= {} Time = {:5.2f}".format(
                        step, float(total_nll), float(total_mse), len(self.model.sampled_weights),
                        t)
                    )

            if step > num_burn_in_steps and (step - num_burn_in_steps) % self.keep_every \
                    == 0:
                weights = tuple(
                    np.asarray(parameter.data.clone().detach().numpy())
                    for parameter in self.model.parameters()
                )

                self.model_wrapper.sampled_weights.append(weights)

    def _instantiate_train_loader(self, x_train, y_train, batch_size):
        """This method basically tries to copy the original training process and insert some
        custom translation of the data to meet the interface of the original code.

        returns trainloader

        Note:
        x_train: input training datapoints. (config + fidelity level)
        y_train: input training targets. (perofmrance of config at that fidleity level)
        """

        self.num_datapoints, self.input_dimensionality = x_train.shape
        logging.debug(
            "Processing %d training datapoints "
            " with % dimensions each." % (self.num_datapoints, self.input_dimensionality)
        )
        assert self.batch_size >= 1, "Invalid batch size. Batches must contain at least a single " \
                                     "sample."
        assert len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[
            1] == 1), "Targets need to be in vector format, i.e (N,) or (N,1)"

        if x_train.shape[0] < self.batch_size:
            logging.warning(
                "Not enough datapoints to form a batch. Use all datapoints in each batch")
            batch_size = x_train.shape[0]

        self.X = x_train
        if len(y_train.shape) == 2:
            self.y = y_train[:, 0]
        else:
            self.y = y_train

        if self.do_normalize_input:
            logging.debug(
                "Normalizing training datapoints to "
                " zero mean and unit variance."
            )

            x_train_, self.x_mean, self.x_std = self.normalize_input(x_train)

            # print(f'x_train_ {x_train_}, x_mean {self.x_mean}, x_std {self.x_std}')
            if self.model_wrapper.use_double_precision:
                x_train_ = torch.from_numpy(x_train_).double()
            else:
                x_train_ = torch.from_numpy(x_train_).float()
        else:
            if self.model_wrapper.use_double_precision:
                x_train_ = torch.from_numpy(x_train).double()
            else:
                x_train_ = torch.from_numpy(x_train).float()

        if self.do_normalize_output:
            logging.debug("Normalizing training labels to zero mean and unit variance.")
            # print(self.y)
            y_train_, self.y_mean, self.y_std = self.model_wrapper.normalize_output(self.y)

            if self.model_wrapper.use_double_precision:
                y_train_ = torch.from_numpy(y_train_).double()
            else:
                y_train_ = torch.from_numpy(y_train_).float()
        else:
            if self.model_wrapper.use_double_precision:
                y_train_ = torch.from_numpy(y_train).double()
            else:
                y_train_ = torch.from_numpy(y_train).float()

        # generate a new trainloader
        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                data_utils.TensorDataset(x_train_, y_train_),
                batch_size=self.batch_size,
                shuffle=True
            )
        )

        self.is_trained = True
        return train_loader, x_train_, y_train_

    def _parse_single_dataset(self, dataset, dataset_id):

        learning_curves, dataset_meta_features, algo_meta_features = \
            dataset.learning_curves, dataset.meta_dataset, dataset.meta_algo

        n_fidelity = learning_curves.shape[-1]
        fidelities = torch.linspace(0., 1., n_fidelity)
        # fixme: is this correct: the model_wrapper needs to know at what fidelity value it is,
        #  but the dataloader does not provide such a map: therefore even spacings are assumed
        n_algos = algo_meta_features.shape[0]

        # filtering out constant columns (which throw an error in normalization)
        keep_cols = torch.std(algo_meta_features.transformed_df, 0) != 0

        # prepare x_train
        fid = fidelities.repeat(n_algos).view(-1, 1)
        x_train = torch.repeat_interleave(
            algo_meta_features.transformed_df[:, keep_cols], 51, dim=0
        )
        x_train = torch.cat((x_train, fid), dim=1)

        # fixme: we need to train on a fixed dataset
        # prepare y_train
        y_train = learning_curves.transformed_df[dataset_id].flatten()

        # final fidelity as prompt & target
        x_test = torch.cat((algo_meta_features.transformed_df[:, keep_cols],
                            torch.ones((n_algos, 1))),
                           dim=1)
        y_test = learning_curves.transformed_df[dataset_id, :, -1].flatten()

        # conversion of x_train, y_train, x_test, y_test to numpy
        return x_train.detach().numpy(), y_train.detach().numpy(), x_test.detach().numpy(), \
            y_test.detach().numpy()

    def _instantiate_sampler(self, num_datapoints, input_dimensionality, num_burn_in_steps,
                             continue_training, lr, mdecay, epsilon):
        """

        # COPYED from PYBNN
        Train a BNN using input datapoints `x_train` with corresponding targets `y_train`.

        :param num_burn_in_steps: Number of burn-in steps to perform.
            This value is passed to the given `optimizer` if it supports special
            burn-in specific behavior.
            Networks sampled during burn-in are discarded.
        :param lr: learning rate
        :param mdecay: momemtum decay
        :param epsilon: epsilon for numerical stability
        :param continue_training: defines whether we want to continue from the last training run

        """

        if self.model_wrapper.use_double_precision:
            self.dtype = np.float64
        else:
            self.dtype = np.float32

        if not continue_training:
            logging.debug("Clearing list of sampled weights.")

            self.model_wrapper.sampled_weights.clear()
            if self.model_wrapper.use_double_precision:
                self.model = self.model_wrapper.get_network(
                    input_dimensionality=input_dimensionality).double()
            else:
                self.model = self.model_wrapper.get_network(
                    input_dimensionality=input_dimensionality).float()

            # FIXME: make theses partial and instantiate them using config, then model parameters
            #  and scale_grad can be (optionally) passed once they are computed
            if self.sampling_method == "adaptive_sghmc":
                self.sampler = AdaptiveSGHMC(self.model.parameters(),
                                             scale_grad=self.dtype(num_datapoints),
                                             num_burn_in_steps=num_burn_in_steps,
                                             lr=self.dtype(lr),
                                             mdecay=self.dtype(mdecay),
                                             epsilon=self.dtype(epsilon))
            elif self.sampling_method == "sgld":
                self.sampler = SGLD(self.model.parameters(),
                                    lr=self.dtype(lr),
                                    scale_grad=num_datapoints)
            elif self.sampling_method == "preconditioned_sgld":
                self.sampler = PreconditionedSGLD(self.model.parameters(),
                                                  lr=self.dtype(lr),
                                                  num_train_points=num_datapoints)
            elif self.sampling_method == "sghmc":
                self.sampler = SGHMC(self.model.parameters(),
                                     scale_grad=self.dtype(num_datapoints),
                                     mdecay=self.dtype(mdecay),
                                     lr=self.dtype(lr))

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
            **train_kwargs,
    ):
        train_loss_fn = instantiate(train_loss_fn)['loss_fn']
        self.train(train_loader, epochs, train_loss_fn, **train_kwargs)

        # Test loss for comparison of selectors
        prediction, ground_truth = self.test(test_loader, test_loss_fns)

        return prediction, ground_truth
