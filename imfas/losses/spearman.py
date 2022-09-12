from typing import Callable, Dict

import torch
import torchsort
from torch.nn import Softmax
from torch.nn.modules.loss import _Loss as Loss


class SpearmanLoss(Loss):
    def __init__(
        self, reduction: str = "mean", ranking_fn: Callable = torchsort.soft_rank, ts_kwargs: Dict = {}
    ) -> None:
        super(SpearmanLoss, self).__init__(reduction=reduction)
        self.ranking_fn = ranking_fn
        self.ts_kwargs = ts_kwargs

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # generate soft ranks
        # FIXME: @Aditya, what is the expected dtype of both tensors?
        input = self.ranking_fn(input, **self.ts_kwargs)
        target = self.ranking_fn(target, **self.ts_kwargs)

        # normalize the soft ranks
        input = input - input.mean()
        input = input / input.norm()

        # target = (target - target.min()) / (target.max() - target.min())
        target = target - target.mean()
        target = target / target.norm()

        # compute the correlation
        speark_rank = (input * target).sum()

        # loss is the complement, which needs to be minimized
        return 1 - speark_rank


class WeightedSpearman(Loss):
    def __init__(
        self, reduction: str = "mean", ranking_fn: Callable = torchsort.soft_rank, ts_kwargs: Dict = {}
    ) -> None:
        super(WeightedSpearman, self).__init__(reduction=reduction)
        self.spearman_loss = SpearmanLoss(reduction, ranking_fn=ranking_fn, ts_kwargs=ts_kwargs)
        self.weight_func = Softmax(dim=1)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = self.weight_func(target)
        input = input * weights
        return self.spearman_loss(input, target)


# FIXME: @TIM deprecate this (was used in old sh variant)
# def spear_halve_loss(halving_op, final_fidelity_performance):
#     pred = halving_op.type(torch.float)
#
#     arg_sorted = torch.argsort(final_fidelity_performance, descending=True)
#
#     rank = torch.zeros(len(final_fidelity_performance))
#
#     for i in range(len(final_fidelity_performance)):
#         rank[arg_sorted[i]] = i
#
#     target = rank
#
#     # normalize the soft ranks
#     pred = pred - pred.mean()
#     pred = pred / pred.norm()
#     target = target - target.mean()
#     target = target / target.norm()
#
#     # compute the loss
#     spear = (pred * target).sum()
#     return spear
#     # return 1 - spear
