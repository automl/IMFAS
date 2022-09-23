import warnings
from functools import partial
from typing import List, Callable

import torch

from imfas.trainer.base_trainer import BaseTrainer


class SliceEvaluator(BaseTrainer):
    def __init__(self, model, max_fidelities: List, masking_fn: Callable, *args, **kwargs):
        """
        This BaseTrainer subclass trains as usual, but evaluates the model on a set of slices of
        different length. In particular, it evaluates the model's capability to "forsee" the future
        conditioned on only partial learning curves
        This class is designed to ev aluate the modelduring test time on a fixed
        number of fidelities (i.e. there is a highest accessible fidelity (via masking).

        In particular it is build with successive halving in mind. Consider its definition.
        """
        super().__init__(model, *args, **kwargs)
        self.max_fidelities = max_fidelities
        self.masking_fn = masking_fn

    def evaluate(self, test_loader, valid_loss_fn, fn_name, *args, **kwargs):
        for fidelity in self.max_fidelities:
            test_loader.dataset.masking_fn = partial(self.masking_fn, max_fidelity=fidelity)

            losses = torch.zeros(len(test_loader))
            for i, (X, y) in enumerate(test_loader):
                self.to_device(X)
                self.to_device(y)

                y_hat = self.model.forward(**X)
                losses[i] = valid_loss_fn(y_hat, y["final_fidelity"])

            wandb.log({f"max fidelity: {fn_name}": losses.mean(), 'fidelity': fidelity})

    def run(self, train_loader, test_loader, epochs, log_freq, *args, **kwargs):
        # TODO: check what happens when this condition is not met
        if epochs != log_freq:
            warnings.warn("SliceEvaluator is intended to be run with log_freq == epochs")
        super().run(train_loader, test_loader, epochs=epochs, log_freq=log_freq, *args, **kwargs)


if __name__ == '__main__':
    """
    Experiment to run SH baseline on the fixed fidelity slices (i.e. maximum available fidelity)
    and compare that against a trainable model, that is evaluated on the same maximum slices.
    """

    from torch.utils.data import DataLoader

    from imfas.models.baselines.successive_halving import SuccessiveHalving
    from imfas.losses.spearman import SpearmanLoss
    from imfas.utils.masking import mask_lcs_to_max_fidelity
    from imfas.data.lcbench.example_data import train_dataset, test_dataset

    import wandb

    wandb.init(entity="tnt", mode='online', project='imfas-iclr', job_type='train')

    # (evaluate SH) ---------------------------------------------
    model = SuccessiveHalving(budgets=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51], eta=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # batch=1, because of SH

    sliceevaluator = SliceEvaluator(
        model,
        max_fidelities=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51],
        # creates a learning curve for these fidelities
        masking_fn=mask_lcs_to_max_fidelity
    )

    # sliceevaluator.evaluate(test_loader, valid_loss_fn=SpearmanLoss(), fn_name='spearman')

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # (Train + evaluate another model) ----------------------------
    from imfas.models.imfas_wp import IMFAS_WP
    from imfas.utils.mlp import MLP

    n_algos = 58
    n_meta_features = 107
    model = IMFAS_WP(
        encoder=MLP(hidden_dims=[n_meta_features, 300, 200]),
        decoder=MLP(hidden_dims=[200, n_algos]),
        input_dim=n_algos,
        n_layers=2
    )
    # model = PlackettTest(encoder=MLP(hidden_dims=[n_meta_features, 100, n_algos])) # constant in
    # fidelity
    sliceevaluator = SliceEvaluator(
        model,
        max_fidelities=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 51],
        masking_fn=mask_lcs_to_max_fidelity,
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )

    # NOTICE: for wandb tracking to be sensible, we need to train the model fully first.
    epochs = 1000
    sliceevaluator.run(
        train_loader,
        test_loader,
        train_loss_fn=SpearmanLoss(),
        valid_loss_fns={"spearman": SpearmanLoss()},
        epochs=epochs,  # <----
        log_freq=epochs  # <----
    )

    print('done')
