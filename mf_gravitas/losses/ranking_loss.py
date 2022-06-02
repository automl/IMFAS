import torch 
import numpy as np

import pdb 

from sklearn.metrics import ndcg_score

from torch.nn import Softmax

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
        
        # target = (target - target.min()) / (target.max() - target.min())
        target = target - target.mean()
        target = target / target.norm()

        
        # compute the correlation 
        speark_rank =  (pred * target).sum()

        # loss is the complement, which needs to be minimized
        return 1 - speark_rank   


def weighted_spearman(pred, target, ranking_fn, **ts_kwargs):
        """
        Loss function for the meta-feature ranker


        Args:
            pred: predicted values
            target: target values
            ts_kwargs: keyword arguments for the loss function

        Returns:
            differentiable loss tensor
        """

        m = Softmax(dim=1)
        weights = m(target)

        # generate soft ranks
        pred = pred*weights
        pred = ranking_fn(pred, **ts_kwargs)
        target = ranking_fn(target, **ts_kwargs)
        
        # normalize the soft ranks
        pred = pred - pred.mean()
        pred = pred / pred.norm()
        
        # target = (target - target.min()) / (target.max() - target.min())
        target = target - target.mean()
        target = target / target.norm()

        
        # compute the correlation 
        speark_rank =  (pred * target).sum()

        # loss is the complement, which needs to be minimized
        return 1 - speark_rank  