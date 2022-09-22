import logging
from typing import List, Optional

import math
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SuccessiveHalving(nn.Module):
    def __init__(self, budgets: List, eta: int = 2, device: str = "cpu", ):
        """
        This class assumes a single batch (i.e. a single dataset!)
        :param budgets: List[int] of budget 'labels' to be used in the successive halving run
        (actual budget values available in the learning curve tensor provided during forward).
        :param eta: int, the reduction factor of the budget in each successive halving run.
        """
        super(SuccessiveHalving, self).__init__()

        self.eta = eta
        self.budgets = budgets
        self.device = device

    def plan_budgets(self, budgets: List, mask: torch.Tensor):
        """
        Plan the successive halving run. This method is called once per run.
        Idea is to calculate the schedule dynamically depending on the currently available budget.
        :param budgets: List[int] of budget 'labels' to be used in the successive halving run,
        amounting to the budget; e.g. epochs or dataset subset size.
        :param mask: torch.Tensor (long), the mask of the learning curve tensor.
        indicating which budgets are accessible. Assumes that max budget is available to all algos.

        :return: torch.Tensor (long), the schedule of indices on the learning curve tensor for the
        successive halving run.
        """

        assert len(budgets) == mask.shape[-1], "Budgets and mask do not match!"
        max_fidelity_idx = mask.sum(dim=-1).max().long() - 1
        min_budget = budgets[0]
        max_budget = budgets[max_fidelity_idx]

        # Actual budget schedule (based on min_budget and eta)
        schedule = torch.tensor(
            [min_budget * self.eta ** i for i in
             range(int(math.log(max_budget / min_budget, self.eta)) + 1)]
        )

        # translate the budget schedule to the index of the learning curve
        # i.e. find first surpassing index
        schedule_index = torch.tensor(
            [next(x for x, val in enumerate(budgets) if val >= b)
             for b in schedule]
        )

        # If more budget has been observed, make decision for final fidelity on the basis of
        # the last observed budget. This is used to make sh comparable to other algorithms
        # from the available max fidelity perspective (which likely is most informative).
        if schedule[-1] < max_budget:
            schedule_index = torch.cat(
                (schedule_index, torch.tensor([max_fidelity_idx]))
            )

        return schedule_index

    def forward(
            self,
            learning_curves: torch.Tensor,
            mask: torch.Tensor,
            cost_curves: Optional[torch.Tensor] = None,
    ):
        """
        SH execution.
        :return: torch.Tensor (float), ranking of the algorithms.
        """

        # batch = learning_curves.shape[0]
        n_algos = learning_curves.shape[1]
        algos_idx = torch.arange(n_algos)
        schedule_index = self.plan_budgets(self.budgets, mask)

        # running variables
        n_dead = 0
        rankings_idx = torch.zeros((n_algos,))
        rankings_values = torch.zeros((n_algos,))
        survivors = torch.ones((n_algos), dtype=torch.int64)
        self.elapsed_time = torch.zeros(1)

        # is attribute for debugging (assertion)
        self.schedule_index = self.plan_budgets(self.budgets, mask)

        for level, budget in enumerate(self.schedule_index.tolist(), start=1):
            # number of survivors in this round
            k = max(1, int(n_algos / self.eta ** level))

            # final round to break all ties?
            if level == len(self.schedule_index):
                k = n_algos - n_dead

            # inquired cost  for the next evaluation round
            if cost_curves is not None:
                self.elapsed_time += cost_curves[:, :, budget][0, survivors == 1].sum(dim=0)
            else:
                self.elapsed_time += self.budgets[budget] * torch.sum(survivors)

            logger.debug(f"Budget {budget}: {k} survivors")

            # dead in relative selection (index requires adjustment)
            slice = learning_curves[:, :, budget]
            alive_idx = algos_idx[survivors == 1]
            dead = torch.topk(slice[0, alive_idx], k, dim=0, largest=False)

            # translate to the algorithm index
            new_dead_algo = algos_idx[survivors == 1][dead.indices]
            logger.debug(f"Dead: {new_dead_algo}")

            # change the survivor flag
            survivors[new_dead_algo] = 0
            logger.debug(f"Deceasceds'performances: "
                         f"{learning_curves[:, :, budget][0, new_dead_algo]}")
            logger.debug(f"Survivors'performances: {learning_curves[:, :, budget] * survivors}")

            # BOOKKEEPING: add the (sorted by performance) dead to the ranking
            rankings_idx[n_dead: n_dead + dead.indices.shape[-1]] = new_dead_algo
            rankings_values[n_dead: n_dead + dead.indices.shape[-1]] = dead.values
            n_dead += dead.indices.shape[-1]

            logger.debug(f"rankings: {rankings_idx}")

        return rankings_idx
        # fixme: do we need to sort this ranking? based on the algorithms
        #  original positions.


if __name__ == "__main__":
    import torch

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    # wandb.init()

    # Check Eta schedules and rankings
    # fixme: move to test!
    batch = 1
    n_algos = 10

    budgets = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51]
    fidelities = len(budgets)
    lcs = torch.arange(batch * n_algos * fidelities).view(batch, n_algos, fidelities)
    cost = torch.arange(batch * n_algos * fidelities).view(batch, n_algos, fidelities)
    mask = torch.cat(
        [torch.ones((batch, n_algos, fidelities - 4), dtype=torch.bool),
         torch.zeros((batch, n_algos, 4), dtype=torch.bool)], axis=2)

    # SH with eta=3
    sh = SuccessiveHalving(budgets, 3)

    assert torch.equal(sh.forward(learning_curves=lcs, mask=mask, cost_curves=cost),
                       torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
    assert torch.equal(sh.elapsed_time[0].long(),
                       torch.tensor([0, 11, 22, 33, 44, 55, 66, 77, 88, 99]).sum() + \
                       torch.tensor([35, 46, 57, 68, 79, 90, 101]).sum() + \
                       torch.tensor([50, 61, 72, 83, 94, 105]).sum())
    assert torch.equal(sh.schedule_index, torch.tensor([0, 2, 6]))  # capped at 6 (last budget)
    # because of the mask, which stops at 6 although eta=3 would expect 8 as last budget!
    assert torch.equal(torch.tensor(budgets)[sh.schedule_index],
                       torch.tensor([5, 15, 35]))  # capped at 35

    assert torch.equal(sh.forward(learning_curves=lcs, mask=mask),
                       torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))

    # Same SH, but with eta=2
    sh = SuccessiveHalving(budgets, 2)

    assert torch.equal(sh.forward(learning_curves=lcs, mask=mask),
                       torch.tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]))
    assert torch.equal(sh.schedule_index, torch.tensor([0, 1, 3, 7]))

    # Check the foward call for batched data

    sh = SuccessiveHalving([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)

    lcs = torch.arange(batch * n_algos * fidelities).reshape(batch, n_algos, fidelities)

    rankings = sh(lcs, mask)

    assert torch.equal(
        rankings, torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                               dtype=torch.float)
    )
