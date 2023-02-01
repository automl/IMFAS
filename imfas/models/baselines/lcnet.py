import numpy as np
import torch
import torch.nn as nn

from pybnn.bohamiann import nll
from pybnn.lcnet import LCNet, get_lc_net_architecture


class LCNetWrapper(LCNet):

    def __init__(
            self,
            sampling_method: str = "adaptive_sghmc",
            use_double_precision: bool = True,
            metrics=(nn.MSELoss,),
            likelihood_function=nll,
            print_every_n_steps=100,

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

        self.sampling_method = sampling_method

    @property
    def network_weights(self) -> tuple:
        """
        Extract current network weight values as `np.ndarray`.
        :return: Tuple containing current network weight values
        """
        return tuple(
            np.asarray(parameter.data.clone().detach().numpy())
            for parameter in self.model.parameters()
        )


    # def forward(
    #         self,
    #         learning_curves: torch.Tensor,
    #         mask: torch.Tensor,
    #             dataset_meta_features,
    #         algo_meta_features,
    #         **kwargs
    # ) -> torch.Tensor:
    #
    #     self.parse_data(learning_curves, mask, dataset_meta_features, algo_meta_features)
    #
    #     # FIXME: what is the train test split?
    #
    #     if self.max_fidelity == 0:
    #         return torch.ones(learning_curves.shape[:-1]) * float("nan")
    #     else:
    #         fidelity = self.fidelities[self.max_fidelity].repeat(self.n_algos, 1)
    #
    #         x_train = torch.cat((algo_meta_features[0], fidelity), dim=1).numpy()
    #         y_train = learning_curves[:, :, self.max_fidelity][0].numpy()
    #
    #         self.training_fn(
    #             x_train=x_train,
    #             y_train=y_train,
    #             num_steps=self.num_steps,
    #             num_burn_in_steps=self.num_burn_in_steps,
    #             lr=self.lr
    #         )
    #
    #     # TODO same as x_train but with 1. as fidelity
    #     x_test = torch.cat(
    #         (algo_meta_features, dataset_meta_features, torch.ones_like(fidelity)), dim=1)
    #     x_test = x_test.detach().numpy()
    #
    #     m, v = self.predict(x_test)
    #     # TODO match the ranking for the respective curves!
    #
    #     return m[-1]

    # def train(self):
    #     self.training = True
    #
    # def eval(self):
    #     self.training = False


    #     if hasattr(self, "no_opt"):
    #         delattr(self, "no_opt")  # FIXME: do we also need that for the base class?

    def to(self, device):
        self.device = device


if __name__ == '__main__':

    file = ['lcnet', 'toy'][0]
    # https://github.com/automl/pybnn/blob/master/test/test_lcnet.py
    # https://github.com/automl/pybnn/blob/master/examples/example_lc_extrapolation.py

    if file == 'lcnet':
        def toy_example(t, a, b):
            return (10 + a * np.log(b * t)) / 10. + 10e-3 * np.random.rand()


        observed = 80
        N = 5
        n_epochs = 10
        observed_t = int(n_epochs * (observed / 100.))

        t_idx = np.arange(1, observed_t + 1) / n_epochs
        t_grid = np.arange(1, n_epochs + 1) / n_epochs

        configs = np.random.rand(N, 2)
        learning_curves = [toy_example(t_grid, configs[i, 0], configs[i, 1]) for i in range(N)]

        X_train = None
        y_train = None
        X_test = None
        y_test = None

        for i in range(N):

            x = np.repeat(configs[i, None, :], t_idx.shape[0], axis=0)
            x = np.concatenate((x, t_idx[:, None]), axis=1)

            x_test = np.concatenate((configs[i, None, :], np.array([[1]])), axis=1)

            lc = learning_curves[i][:observed_t]
            lc_test = np.array([learning_curves[i][-1]])

            if X_train is None:
                X_train = x
                y_train = lc
                X_test = x_test
                y_test = lc_test
            else:
                X_train = np.concatenate((X_train, x), 0)
                y_train = np.concatenate((y_train, lc), 0)
                X_test = np.concatenate((X_test, x_test), 0)
                y_test = np.concatenate((y_test, lc_test), 0)

        # result of the following config details how the input is expected
        # observed = 80  --> 80% of the curve is available
        # N = 5
        # n_epochs = 10
        #
        # x_train: [[0.32972551 0.08793102 0.1       ]
        #  [0.32972551 0.08793102 0.2       ]
        #  [0.32972551 0.08793102 0.3       ]
        #  [0.32972551 0.08793102 0.4       ]
        #  [0.32972551 0.08793102 0.5       ]
        #  [0.32972551 0.08793102 0.6       ]
        #  [0.32972551 0.08793102 0.7       ]
        #  [0.32972551 0.08793102 0.8       ]  --> 8 samples (leading with the two config values
        #  + fidelity level) -- meaning here the first curve terminates at 80% of the total epochs
        #  [0.15403017 0.84915016 0.1       ]
        #  [0.15403017 0.84915016 0.2       ]
        #  [0.15403017 0.84915016 0.3       ]
        #  [0.15403017 0.84915016 0.4       ]
        #  [0.15403017 0.84915016 0.5       ]
        #  [0.15403017 0.84915016 0.6       ]
        #  [0.15403017 0.84915016 0.7       ]
        #  [0.15403017 0.84915016 0.8       ] end of the second curve
        #  [0.99019806 0.21469587 0.1       ]
        #  [0.99019806 0.21469587 0.2       ]
        #  [0.99019806 0.21469587 0.3       ]
        #  [0.99019806 0.21469587 0.4       ]
        #  [0.99019806 0.21469587 0.5       ]
        #  [0.99019806 0.21469587 0.6       ]
        #  [0.99019806 0.21469587 0.7       ]
        #  [0.99019806 0.21469587 0.8       ] end of the third curve
        #  [0.94964878 0.00566681 0.1       ]
        #  [0.94964878 0.00566681 0.2       ]
        #  [0.94964878 0.00566681 0.3       ]
        #  [0.94964878 0.00566681 0.4       ]
        #  [0.94964878 0.00566681 0.5       ]
        #  [0.94964878 0.00566681 0.6       ]
        #  [0.94964878 0.00566681 0.7       ]
        #  [0.94964878 0.00566681 0.8       ] end of the fourth curve
        #  [0.68857372 0.37327868 0.1       ]
        #  [0.68857372 0.37327868 0.2       ]
        #  [0.68857372 0.37327868 0.3       ]
        #  [0.68857372 0.37327868 0.4       ]
        #  [0.68857372 0.37327868 0.5       ]
        #  [0.68857372 0.37327868 0.6       ]
        #  [0.68857372 0.37327868 0.7       ]
        #  [0.68857372 0.37327868 0.8       ]], end of the fifth curve
        #
        #  x_test: [[0.32972551 0.08793102 1.        ]  --> final fidelity value for these 5 configs
        #  [0.15403017 0.84915016 1.        ]
        #  [0.99019806 0.21469587 1.        ]
        #  [0.94964878 0.00566681 1.        ]
        #  [0.68857372 0.37327868 1.        ]]
        #
        # y_train.shape: (40,), y_test.shape: (5,),
        #
        # y_train:
        # [0.85346409 0.87631892 0.88968814 0.89917376
        #  0.90653137 0.91254297 0.91762572 0.92202859 # end of the first curve
        #  0.96587761 0.97655417 0.98279956 0.98723073
        #  0.99066781 0.99347612 0.9958505  0.99790729 # end of the second curve
        #  0.62845162 0.69708692 0.737236   0.76572222
        #  0.78781785 0.8058713  0.82113527 0.83435752 # end of the third curve
        #  0.2972352  0.36305984 0.40156478 0.42888447
        #  0.45007527 0.46738942 0.48202832 0.49470911 # end of the fourth curve
        #  0.77930875 0.82703705 0.85495631 0.87476534
        #  0.89013042 0.9026846  0.91329901 0.92249363], # end of the fifth curve
        #
        #  actual final fidelity values for the 5 configs of test:
        # y_test: [0.9293862  1.00134437 0.85645315 0.51589991 0.93785871]

        print(f'X_train.shape: {X_train.shape}, x_test.shape: {X_test.shape}, \nx_train:'
              f' {X_train},\n x_test: {X_test}')
        print(
            f'y_train.shape: {y_train.shape}, y_test.shape: {y_test.shape},\ny_train: {y_train}, \ny_test: {y_test}')
        model = LCNet()

        model.train(X_train, y_train, num_steps=500, num_burn_in_steps=40, lr=1e-2)

        m, v = model.predict(X_test)

    elif file == 'toy':

        import numpy as np
        import time
        import matplotlib.pyplot as plt
        from pybnn.lc_extrapolation.learning_curves import MCMCCurveModelCombination

        observed = 40
        n_epochs = 100

        t_idx = np.arange(1, observed + 1)
        t_idx_full = np.arange(1, n_epochs + 1)


        def toy_example(t, a, b):
            return (10 + a * np.log(b * t + 1e-8)) / 10. + 10e-3 * np.random.rand()


        a = np.random.rand()
        b = np.random.rand()
        lc = [toy_example(t / n_epochs, a, b) for t in t_idx_full]

        model = MCMCCurveModelCombination(n_epochs + 1,
                                          nwalkers=50,
                                          nsamples=800,
                                          burn_in=500,
                                          recency_weighting=False,
                                          soft_monotonicity_constraint=False,
                                          monotonicity_constraint=True,
                                          initial_model_weight_ml_estimate=True)
        st = time.time()
        model.fit(t_idx, lc[:observed])
        print("Training time: %.2f" % (time.time() - st))

        st = time.time()
        p_greater = model.posterior_prob_x_greater_than(n_epochs + 1, .5)
        print("Prediction time: %.2f" % (time.time() - st))

        m = np.zeros([n_epochs])
        s = np.zeros([n_epochs])

        for i in range(n_epochs):
            p = model.predictive_distribution(i + 1)
            m[i] = np.mean(p)
            s[i] = np.std(p)

        mean_mcmc = m[-1]
        std_mcmc = s[-1]

        plt.plot(t_idx_full, m, color="purple", label="LC-Extrapolation")
        plt.fill_between(t_idx_full, m + s, m - s, alpha=0.2, color="purple")
        plt.plot(t_idx_full, lc)

        plt.xlim(1, n_epochs)
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel("Validation error")
        plt.axvline(observed, linestyle="--", color="black")
        plt.show()
