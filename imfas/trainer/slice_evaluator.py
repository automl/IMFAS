from typing import List

from functools import partial

import torch

from imfas.trainer.base_trainer import BaseTrainer


class SliceEvaluator(BaseTrainer):
    def __init__(self, model, max_fidelities: List, masking_fn=None, *args, **kwargs):
        """
        This class is designed to evaluate the model during test time on a fixed
        number of fidelities (i.e. there is a highest accessible fidelity (via masking).

        In particular it is build with successive halving in mind. Consider its definition.
        """
        super().__init__(model, *args, **kwargs)
        self.max_fidelities = max_fidelities
        self.masking_fn = masking_fn

    def evaluate(self, test_loader, valid_loss_fn, *args, **kwargs):

        n_algos = len(test_loader.dataset.meta_algo)
        if hasattr(self.model, "budgets"):
            # SUCCESSIVE HALVING speciality
            budgets = self.model.budgets

        for fidelity in self.max_fidelities:

            if hasattr(self.model, "budgets"):
                # SUCCESSIVE HALVING speciality
                self.model.__init__(budgets, eta=self.model.eta)

            test_loader.dataset.masking_fn = partial(self.masking_fn, max_fidelity=fidelity)

            losses = torch.zeros(n_algos)

            for X, y in test_loader:
                self.to_device(X)
                self.to_device(y)

                y_hat = self.model.forward(**X)
                losses.append(valid_loss_fn(y_hat, y["final_fidelity"]))

            return losses


class SliceEvaluatorSH(SliceEvaluator):
    def train(self, *args, **kwargs):
        return None
