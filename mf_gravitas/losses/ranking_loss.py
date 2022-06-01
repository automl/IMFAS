import torch


def spearman(pred, target, ranking_fn, **ts_kwargs):
    """
    Loss function for the meta-feature ranker


    Args:
        pred: predicted values
        target: target values
        ts_kwargs: keyword arguments for the loss function

    Returns:
        differentiable loss tensor
    """

    # generate soft ranks
    pred = ranking_fn(pred, **ts_kwargs)
    target = ranking_fn(target, **ts_kwargs)

    # normalize the soft ranks
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()

    # compute the loss
    spear_loss = (pred * target).sum()

    return 1 - spear_loss


def spear_halve_loss(halving_op, final_fidelity_performance):
    pred = halving_op.type(torch.float)

    arg_sorted = torch.argsort(final_fidelity_performance)

    rank = torch.zeros(len(final_fidelity_performance))

    for i in range(len(final_fidelity_performance)):
        rank[arg_sorted[i]] = i

    target = rank

    # normalize the soft ranks
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()

    # compute the loss
    spear = (pred * target).sum()

    return 1 - spear
