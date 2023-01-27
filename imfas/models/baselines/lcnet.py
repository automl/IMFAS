import logging
import time
from itertools import islice

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from pybnn.bohamiann import nll
from pybnn.lcnet import LCNet, get_lc_net_architecture
from pybnn.priors import weight_prior, log_variance_prior
from pybnn.sampler import AdaptiveSGHMC, SGLD, SGHMC, PreconditionedSGLD
from pybnn.util.infinite_dataloader import infinite_dataloader
from pybnn.util.normalization import zero_mean_unit_var_denormalization
from scipy.stats import norm


class LCNetWrapper(LCNet):

    def __init__(
            self,
            sampling_method: str = "adaptive_sghmc",
            use_double_precision: bool = True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=100,
            num_steps=500,
            num_burn_in_steps=40,
            lr=1e-2

    ) -> None:

        super(LCNet, self).__init__(
            get_network=get_lc_net_architecture,
            normalize_input=True,
            normalize_output=False,
            sampling_method=sampling_method,
            use_double_precision=use_double_precision,
            metrics=metrics,
            likelihood_function=likelihood_function,
            print_every_n_steps=print_every_n_steps
        )

        print("nll", nll, metrics)
        self.no_opt = True  # FIXME: @Aditya - this hack should not be necessary.
        self.num_steps = num_steps
        self.num_burn_in_steps = num_burn_in_steps
        self.lr = lr

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor,
                dataset_meta_features, algo_meta_features, **kwargs) -> \
            torch.Tensor:

        self.max_fidelity = int(mask.sum(dim=-1).max().item())
        self.n_algos = learning_curves.shape[1]

        # fixme: is this correct: the model needs to know at what fidelity value it is,
        #  but the dataloader does not provide such a map: therefore even spacings are assumed
        self.fidelities = torch.linspace(0., 1., learning_curves.shape[-1])

        if self.max_fidelity == 0:
            return torch.ones(learning_curves.shape[:-1]) * float("nan")
        else:
            fidelity = self.fidelities[self.max_fidelity].repeat(self.n_algos, 1)

            x_train = torch.cat((algo_meta_features[0], fidelity), dim=1).numpy()
            y_train = learning_curves[:, :, self.max_fidelity][0].numpy()

            self.training_fn(
                x_train=x_train,
                y_train=y_train,
                num_steps=self.num_steps,
                num_burn_in_steps=self.num_burn_in_steps,
                lr=self.lr
            )

        # TODO same as x_train but with 1. as fidelity
        x_test = torch.cat(
            (algo_meta_features, dataset_meta_features, torch.ones_like(fidelity)), dim=1)
        x_test = x_test.detach().numpy()

        m, v = self.predict(x_test)
        # TODO match the ranking for the respective curves!

    def train(self):
        self.training = True

    def eval(self):
        self.training = False
        if hasattr(self, "no_opt"):
            delattr(self, "no_opt")  # FIXME: do we also need that for the base class?

    def to(self, device):
        self.device = device

    def training_fn(self, x_train: np.ndarray, y_train: np.ndarray,
                    num_steps: int = 13000,
                    keep_every: int = 100,
                    num_burn_in_steps: int = 3000,
                    lr: float = 1e-2,
                    batch_size=20,
                    epsilon: float = 1e-10,
                    mdecay: float = 0.05,
                    continue_training: bool = False,
                    verbose: bool = False,
                    **kwargs):

        """

        # COPYED from PYBNN
        Train a BNN using input datapoints `x_train` with corresponding targets `y_train`.
        :param x_train: input training datapoints.
        :param y_train: input training targets.
        :param num_steps: Number of sampling steps to perform after burn-in is finished.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param keep_every: Number of sampling steps (after burn-in) to perform before keeping a sample.
            In total, `num_steps // keep_every` network weights will be sampled.
        :param num_burn_in_steps: Number of burn-in steps to perform.
            This value is passed to the given `optimizer` if it supports special
            burn-in specific behavior.
            Networks sampled during burn-in are discarded.
        :param lr: learning rate
        :param batch_size: batch size
        :param epsilon: epsilon for numerical stability
        :param mdecay: momemtum decay
        :param continue_training: defines whether we want to continue from the last training run
        :param verbose: verbose output
        """
        logging.debug("Training started.")
        start_time = time.time()

        num_datapoints, input_dimensionality = x_train.shape
        logging.debug(
            "Processing %d training datapoints "
            " with % dimensions each." % (num_datapoints, input_dimensionality)
        )
        assert batch_size >= 1, "Invalid batch size. Batches must contain at least a single sample."
        assert len(y_train.shape) == 1 or (len(y_train.shape) == 2 and y_train.shape[
            1] == 1), "Targets need to be in vector format, i.e (N,) or (N,1)"

        if x_train.shape[0] < batch_size:
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
            if self.use_double_precision:
                x_train_ = torch.from_numpy(x_train_).double()
            else:
                x_train_ = torch.from_numpy(x_train_).float()
        else:
            if self.use_double_precision:
                x_train_ = torch.from_numpy(x_train).double()
            else:
                x_train_ = torch.from_numpy(x_train).float()

        if self.do_normalize_output:
            logging.debug("Normalizing training labels to zero mean and unit variance.")
            y_train_, self.y_mean, self.y_std = self.normalize_output(self.y)

            if self.use_double_precision:
                y_train_ = torch.from_numpy(y_train_).double()
            else:
                y_train_ = torch.from_numpy(y_train_).float()
        else:
            if self.use_double_precision:
                y_train_ = torch.from_numpy(y_train).double()
            else:
                y_train_ = torch.from_numpy(y_train).float()

        # had to add in
        x_train_.requires_grad = True
        y_train_.requires_grad = True

        train_loader = infinite_dataloader(
            data_utils.DataLoader(
                data_utils.TensorDataset(x_train_, y_train_),
                batch_size=batch_size,
                shuffle=True
            )
        )

        if self.use_double_precision:
            dtype = np.float64
        else:
            dtype = np.float32

        if not continue_training:
            logging.debug("Clearing list of sampled weights.")

            self.sampled_weights.clear()
            if self.use_double_precision:
                self.model = self.get_network(input_dimensionality=input_dimensionality).double()
            else:
                self.model = self.get_network(input_dimensionality=input_dimensionality).float()

            if self.sampling_method == "adaptive_sghmc":
                self.sampler = AdaptiveSGHMC(self.model.parameters(),
                                             scale_grad=dtype(num_datapoints),
                                             num_burn_in_steps=num_burn_in_steps,
                                             lr=dtype(lr),
                                             mdecay=dtype(mdecay),
                                             epsilon=dtype(epsilon))
            elif self.sampling_method == "sgld":
                self.sampler = SGLD(self.model.parameters(),
                                    lr=dtype(lr),
                                    scale_grad=num_datapoints)
            elif self.sampling_method == "preconditioned_sgld":
                self.sampler = PreconditionedSGLD(self.model.parameters(),
                                                  lr=dtype(lr),
                                                  num_train_points=num_datapoints)
            elif self.sampling_method == "sghmc":
                self.sampler = SGHMC(self.model.parameters(),
                                     scale_grad=dtype(num_datapoints),
                                     mdecay=dtype(mdecay),
                                     lr=dtype(lr))

        batch_generator = islice(enumerate(train_loader), num_steps)

        for step, (x_batch, y_batch) in batch_generator:
            self.sampler.zero_grad()
            loss = self.likelihood_function(input=self.model(x_batch), target=y_batch)
            # Add prior. Note the gradient is computed by: g_prior + N/n sum_i grad_theta_xi see Eq 4
            # in Welling and Whye The 2011. Because of that we divide here by N=num of datapoints since
            # in the sample we rescale the gradient by N again
            loss -= log_variance_prior(self.model(x_batch)[:, 1].view((-1, 1))) / num_datapoints
            loss -= weight_prior(self.model.parameters(), dtype=dtype) / num_datapoints
            loss.backward()
            self.sampler.step()

            if verbose and step > 0 and step % self.print_every_n_steps == 0:

                # compute the training performance of the ensemble
                if len(self.sampled_weights) > 1:
                    mu, var = self.predict(x_train)
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
                          "Samples= {} Time = {:5.2f}".format(step,
                                                              float(total_nll),
                                                              float(total_mse),
                                                              len(self.sampled_weights), t))

            if step > num_burn_in_steps and (step - num_burn_in_steps) % keep_every == 0:
                weights = self.network_weights

                self.sampled_weights.append(weights)

        self.is_trained = True
