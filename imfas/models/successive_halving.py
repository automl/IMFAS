import logging
from typing import List

import math
import torch
import torch.nn as nn
import wandb

logger = logging.getLogger(__name__)


# TODO write a test ensuring that each algorithm is ranked only once!

class SuccessiveHalving(nn.Module):
    """Non-parametric, myopic, learning curve aware Algorithm Selector."""

    def __init__(self, budgets: List[int], eta: int, device: str = 'cpu',
                 budget_continue: bool = False):
        """
        :param budgets: List[int] of budgets to be used in the successive halving run (actual
        budget values available in the learning curve tensor provided during forward).
        :param eta: int, the reduction factor of the budget in each successive halving run.
        :param device: str, the device to run the algorithm on.
        :param budget_continue: bool, whether we would continue the training of an algorithm or
        have to retrain it for the next budget level.
        """
        super(SuccessiveHalving, self).__init__()
        self.budgets = budgets
        self.min_budget = budgets[0]
        self.max_budget = budgets[-1]
        self.eta = eta
        self.depletion_fn = [self.depleted_budget_retrain, self.depleted_budget_continuation][
            budget_continue]

        self.device = device

        # Actual budget schedule (based on min_budget and eta)
        schedule = torch.tensor(
            [self.min_budget * eta ** i for i in
             range(int(math.log(self.max_budget / self.min_budget, eta)) + 1)])

        # translate the budget schedule to the index of the learning curve
        # (find first surpassing index)
        self.schedule_index = torch.tensor(
            [(b <= torch.tensor(budgets)).nonzero(as_tuple=True)[0][0].tolist()
             for b in schedule])

    def forward(self, learning_curves: torch.Tensor, mask: torch.Tensor, *args, **kwargs):
        # wrapper to hardcode the mask into the data!

        # determine max available fidelity and index not mask!
        # assuming that mask is basically stack of a torch.ones & torch.zero tensor.
        max_fidelity_idx = mask.sum(dim=-1).max()
        logger.debug(f"Max fidelity: {max_fidelity_idx}")

        # since masking might disrupt the schedule, we need to adjust the schedule
        # i.e. capping and defaulting to the max budget available given the mask.
        self.schedule_index = self.schedule_index[self.schedule_index < max_fidelity_idx]

        if self.schedule_index[-1] < max_fidelity_idx:
            self.schedule_index = torch.cat(
                (self.schedule_index, torch.tensor([max_fidelity_idx - 1])))

        self.schedule_index = self.schedule_index.tolist()
        print(self.schedule_index)
        # FIXME: instead of this hack, make the indexing of _forward appropriate
        if len(learning_curves.shape) == 3:
            # batched
            rankings = torch.zeros((learning_curves.shape[0], learning_curves.shape[1]))
            for i in range(learning_curves.shape[0]):
                rankings[i] = self._forward(
                    learning_curves[i, :, :max_fidelity_idx].view(1, -1, max_fidelity_idx)
                )
            return rankings

        else:
            return self._forward(learning_curves[:, :max_fidelity_idx])

    def depleted_budget_retrain(self, budget_idx: List[int], survivors):
        """
        Return the actual budget that was used in this iteration, assuming,
        that we need to retrain the model from scratch.
        :param budget_idx: List[int], of len 2 the budget indices of the previous and current
        budget.
        """
        return self.budgets[budget_idx[-1]] * survivors.sum(dim=1).max()

    def depleted_budget_continuation(self, budget_idx: List[int], survivors):
        """
        Return the actual budget that was used in this iteration, assuming,
        that we can continue training the model.
        :param budget_idx: List[int], of len 2 the budget indices of the previous and current
        """
        return (self.budgets[budget_idx[0]] - budgets[budget_idx[1]]) * survivors.sum(dim=1).max()

    def _forward(self, learning_curves: torch.Tensor):
        """
        :param learning_curves: torch.Tensor of shape [batch, n_algos, n_fidelities] containing
        the learning curves of the algorithms. SH assumes that the fidelity levels
        are defined at linear budget intervals. This means, that the tensor contains
        the entire learning curve, and we skip levels of budgets according to eta.
        :return: torch.Tensor[int] of shape [batch, n_algos] A vector representing the
        ranking of the algorithms with 0 being the lowest.
        """
        depleted_budget = 0  # amount of budget used for the entire sh run

        # FIXME: check for batch (but optional)
        batch = learning_curves.shape[0]
        n_algos = learning_curves.shape[1]

        rankings_idx = torch.zeros((n_algos,))
        rankings_values = torch.zeros((n_algos,))

        algos_idx = torch.arange(n_algos).view(1, -1)
        survivors = torch.ones((batch, n_algos), dtype=torch.bool)
        n_dead = 0

        logger.debug(f"Budgets: {self.budgets}")
        logger.debug(f"Schedule: {self.schedule_index}")
        logger.debug(f"Learning curves: \n{learning_curves}")

        for i, budget in enumerate(self.schedule_index, start=1):
            # number of survivors in this round
            k = max(1, int(n_algos / self.eta ** i))

            depleted_budget += self.depletion_fn([budget - 1, budget], survivors)

            logger.debug(f"Budget {budget + 1}: {k} survivors")

            # dead in relative selection (index requires adjustment)
            dead = torch.topk(learning_curves[:, :, budget][survivors], k, dim=0, largest=False)

            # translate to the algorithm index
            new_dead_algo = algos_idx[survivors][dead.indices]
            logger.debug(f"Dead: {new_dead_algo}")

            # change the survivor flag
            survivors[:, new_dead_algo] = False
            logger.debug(f"Survivors: {learning_curves[:, :, budget] * survivors}")

            # add the (sorted by performance) dead to the ranking
            rankings_idx[n_dead: n_dead + dead.indices.shape[-1]] = new_dead_algo
            rankings_values[n_dead: n_dead + dead.indices.shape[-1]] = dead.values
            n_dead += dead.indices.shape[-1]

            logger.debug(f"rankings: {rankings_idx}")

        # resolve tie on the final ranking
        if n_dead < n_algos:
            # sort the remaining by performance. (starting at the worst)
            remaining = survivors.sum()
            dead = torch.topk(learning_curves[:, :, budget][survivors], k=remaining, dim=0,
                              largest=False)

            rankings_idx[n_dead:] = algos_idx[survivors][dead.indices]
            rankings_values[n_dead: n_dead + dead.indices.shape[-1]] = dead.values

            depleted_budget += self.depletion_fn([budget - 1, budget], survivors)
            logger.debug(f"rankings: {rankings_idx}")
            logger.debug(f"rankings' values: {rankings_values}")

        wandb.log({"depleted_budget": depleted_budget})

        return rankings_idx


if __name__ == '__main__':
    import torch

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    wandb.init()
    # Check Eta schedules and rankings
    # fixme: move to test!
    budgets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51]
    l = torch.arange(7 * 11).view(1, 7, 11)

    # SH with eta=3
    sh = SuccessiveHalving(budgets, 3)
    assert torch.equal(sh.schedule_index, torch.tensor([0, 2, 8]))
    assert torch.equal(torch.tensor(budgets)[sh.schedule_index], torch.tensor([5, 15, 45]))
    assert torch.equal(sh._forward(learning_curves=l), torch.tensor([0., 1., 2., 3., 4., 5., 6.]))

    # Same SH, but with eta=2
    sh = SuccessiveHalving(budgets, 2)
    assert torch.equal(sh.schedule_index, torch.tensor([0, 1, 3, 7]))
    assert torch.equal(sh._forward(learning_curves=l), torch.tensor([0., 1., 2., 3., 4., 5., 6.]))

    # Check the foward call for batched data

    sh = SuccessiveHalving([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)

    batch = 2
    n_algos = 10
    fidelities = len(sh.budgets)
    lcs = torch.arange(batch * n_algos * fidelities).reshape(batch, n_algos, fidelities)

    rankings = sh(lcs, mask=torch.cat(
        [torch.ones(n_algos - 4, dtype=torch.bool),
         torch.zeros(4, dtype=torch.bool)]).view(1, -1))

    assert torch.equal(
        rankings, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.float))
