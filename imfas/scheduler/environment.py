from collections import namedtuple

import torch


class Environment:
    def __init__(self, cost_curves, validation_curves, test_curves, regret_model, regret_fn):
        """
        Rl environment for the imfas scheduler.
        Actionspace are the fidelities.

        Parameters
        ----------
        cost_curves : torch.Tensor of shape (n_datasets, n_learners, n_fidelities)
        """
        self.cost_curves = cost_curves
        self.validation_curves = validation_curves
        self.test_curves = test_curves

        self.n_datasets, self.n_learners, self.n_fidelities = cost_curves.shape
        self.regret_model = regret_model
        self.regret_model.eval()

        self.regret_fn = regret_fn

        self.state = None
        self.action = None
        self.reward = None
        self.done = False
        self.info = None

        self.incured_costs = 0
        self.observed_fidelities = torch.zeros_like(cost_curves)

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def step(self, action):
        regret_fn()
        self.regret_model()

        # update state

        return reward, self.state, done

    def reset(self):
        self.incured_costs = 0

        # initialize state
        return self.state
