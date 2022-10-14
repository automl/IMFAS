import logging
from abc import abstractmethod
from functools import partial
from typing import List, Callable

import numpy as np
import scipy as sp
import scipy.optimize
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# FIXME: move to utils & make SH + algo selection baseline inherit from this
class ModelInterface:

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass


class ParametricLC(ModelInterface):
    functionals = {
        'pow2': lambda x, a, b: -a * x ** (-b),
        'pow3': lambda x, a, b, c: a - b * x ** (-c),
        'log2': lambda x, a, b: -a * np.log(x) + b,
        'exp3': lambda x, a, b, c: a * np.exp(-b * x) + c,
        'exp2': lambda x, a, b: a * np.exp(-b * x),
        'lin2': lambda x, a, b: a * x + b,
        'vap3': lambda x, a, b, c: np.exp(a + b / x + c * np.log(x)),
        'mmf4': lambda x, a, b, c, d: (a * b + c * x ** d) / (b + x ** d),
        'wbl4': lambda x, a, b, c, d: (c - b * np.exp(-a * (x ** d))),
        'exp4': lambda x, a, b, c, d: c - np.exp(-a * (x ** d) + b),
        'expp3': lambda x, a, b, c: c - np.exp((x - b) ** a),
        # fun = lambda x: a * np.exp(-b*x) + c

        'pow4': lambda x, a, b, c, d: a - b * (x + d) ** (-c),  # has to closely match pow3,
        'ilog2': lambda x, a, b: b - (a / np.log(x)),
        'expd3': lambda x, a, b, c: c - (c - a) * np.exp(-b * x),
        'logpower3': lambda x, a, b, c: a / (1 + (x / np.exp(b)) ** c),
        'last1': lambda x, a: (a + x) - x  # casts the prediction to have the correct size
    }

    def __init__(self, function: str, budgets: List[int]):
        """
        :param function: str, the name of the function to use for the parametric learning curve.
        """
        assert function in self.functionals.keys(), f"Function {function} not implemented!"
        self.function = function
        self.functional = self.functionals[function]
        self.budgets = np.array(budgets, dtype=np.float64)

    def __repr__(self):
        return f"Parametric_LC({self.function})"

    @property
    def n_parameters(self) -> int:
        return self.functional.__code__.co_argcount - 1

    @property
    def parameter_names(self):
        return ['a', 'b', 'c', 'd'][0:self.n_parameters]

    def objective_factory(self, x, y) -> Callable:
        """
        wrapper for the objective function to be minimized. Conditions on the
        x values and returns
        """
        f = partial(self.functional, x=x)

        def objective(params):
            """params: iterable of parameter values to be optimized"""
            return (f(**dict(zip(self.parameter_names, params))) - y)  # ** 2

        return objective

    def fit(self, x: np.ndarray, Y: np.ndarray, restarts: int = 10) -> None:
        """
        :param x: 1D np.ndarray, the budgets on which the learning curves are observed.
        :param Y: 2D np.ndarray, the learning curves observed on the budgets for the respective
        algorithms on the same dataset.
        :param restarts: int, the number of restarts to use for the optimization.
        only the best result (least cost) is kept.
        """

        # the best parameters for each learning curve
        parameters_lc = np.zeros((restarts, Y.shape[1], self.n_parameters))

        # fixme: currently assuming batch == 1
        cost = np.zeros((restarts, Y.shape[1]))  # * np.inf
        #  multiple reinitializations for stability
        for r in tqdm(range(restarts)):
            init = np.random.rand(Y.shape[1], self.n_parameters)

            for i in range(Y.shape[1]):
                try:
                    parameters = sp.optimize.least_squares(
                        self.objective_factory(x, Y[:, i, :][0]),
                        init[i],
                        method='lm'
                    )
                    parameters_lc[r, i, :] = parameters.x
                    cost[r, i] = parameters.cost
                except Exception as e:
                    logger.error(
                        f"Error in fitting: {e} for function {self.function} at init {init[i]}")
                    parameters_lc[r, i, :] = np.nan
                    cost[r, i] = np.inf
            # if there is at least one cost row that is not nan, we can break
            # if r > 10 and not np.isinf(cost[:r]).all():
            #     break

        if np.all(np.isnan(cost)):
            logger.error(f"Could not fit any learning curve for function {self.function}")

        # find the best parameters for each learning curve by cost & associated cost
        self.parameters_lc = parameters_lc[np.argmin(cost, axis=0), np.arange(cost.shape[1]), :]
        self.cost = cost[np.argmin(cost, axis=0), np.arange(cost.shape[1])]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.functional(x, *params) for params in self.parameters_lc])

    def forward(self, lc_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param lc_tensor: torch.Tensor, the tensor containing the learning curves observed on a
        single dataset.
        :param mask: torch.Tensor, the mask indicating how far the learning curves are observed.
        :return: torch.Tensor, ranking score (i.e. performance for each algorithm observed
        at maximum fidelity
        """
        self.max_fidelity = mask.sum(dim=-1).max().item()

        # fit the parametric learning curve to the data.
        self.fit(self.budgets, lc_tensor[:, :, :self.max_fidelity].cpu().numpy())

        # Predict for every curve on max-fidelity (extrapolation)
        return torch.tensor(self.predict(self.budgets[self.max_fidelity - 1]))

    def plot_curves(self, x, y, ax):
        y_hats = self.predict(x)
        for y, y_hat in zip(y[0], y_hats):
            ax.plot(x, y_hat, color='red', alpha=0.5, linewidth=1., label='fitted lc')
            ax.plot(x, y, color='grey', alpha=0.5, linewidth=0.5, label='actual lc')
        ax.set_title(self.function)
        ax.set_xlabel('Budget')
        ax.set_ylabel('Performance')

    def sample_lc(self):
        pass


class BestParametricLC(ModelInterface):
    def __init__(self, budgets: List[int]):
        self.budgets = np.array(budgets, dtype=np.float64)
        self.parametric_lcs = {
            name: ParametricLC(name, budgets)
            for name in ParametricLC.functionals.keys()
        }

    def fit(self, x: np.ndarray, Y: np.ndarray, restarts: int = 10) -> None:
        for parametric_lc in self.parametric_lcs.values():
            parametric_lc.fit(x, Y, restarts)

        # find the best parametric learning curve for each learning curve by cost
        self.costs = np.array(
            [parametric_lc.cost for parametric_lc in self.parametric_lcs.values()])

        # find out which parametric lc is the best for which algorithm
        self.curve_name = np.array(list(self.parametric_lcs.keys()))[np.argmin(self.costs, axis=0)]
        self.curve = np.argmin(self.costs, axis=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array(
            [parametric_lc.predict(x)
             for parametric_lc in self.parametric_lcs.values()]
        )
        predictions[self.curve]

    def forward(self, lc_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.max_fidelity = mask.sum(dim=-1).max().item()

        # fit the parametric learning curve to the data.
        self.fit(self.budgets, lc_tensor[:, :, :self.max_fidelity].cpu().numpy())


if __name__ == '__main__':
    from imfas.data.lcbench.example_data import train_dataset

    lc_predictor = ParametricLC('exp3', budgets=list(range(0, 51)))
    print(lc_predictor.n_parameters)

    # Batched learning curves (batch_size, n_algorithms, n_budgets)
    X, y = train_dataset[3]
    lc_tensor = X['learning_curves']
    shape = lc_tensor.shape
    lc_tensor = lc_tensor.view(1, *shape)
    mask = X['mask'].view(1, *shape)
    ranking = y['final_fidelity'].view(1, shape[0])

    # print(lc_predictor.forward(lc_tensor, mask))

    # lc_predictor.plot_curves(x=lc_predictor.budgets, y=lc_tensor, ax=plt.gca())
    # plt.show()

    # BestParametricLC(lc_predictor.budgets).fit(lc_predictor.budgets, lc_tensor.cpu().numpy())

    lc_predictor = BestParametricLC.forward(lc_predictor.budgets)

    print()
