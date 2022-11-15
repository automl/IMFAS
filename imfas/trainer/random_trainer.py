import wandb
from hydra.utils import instantiate
from tqdm import tqdm
import torch

from imfas.trainer.base_trainer import BaseTrainer


class RandomTrainer(BaseTrainer):
    def __init__(self, model, reps, **kwargs):

        self.reps = reps
        self.model = model
        self.device = self.model.device
        self._step = 0

    def train(self, train_loader, epochs, loss_fn, log_freq=1) -> None:
        """define one epoch of training"""
        pass

    def validate(self, valid_loader, valid_loss_fn, function_name):
        pass

    def test(self, test_loader, test_loss_fns, **kwargs):
        prediction = []
        ground_truth = []
        test_loss_fns = {f_name: instantiate(loss_fn) for f_name, loss_fn in test_loss_fns.items()}
        print(f'test_loss_fns: {test_loss_fns}')
        for (X, y) in tqdm(test_loader, desc=f'Test Dataset i'):
            self.to_device(X)
            self.to_device(y)

            for reps in tqdm(range(self.reps), desc='random_reps'):
                y_hat = self.model.forward(**X)

                for f_name, loss_fn in test_loss_fns.items():
                    # loss_fn = instantiate(loss_fn)

                    loss = loss_fn(y_hat, y["final_fidelity"])

                    # track with wandb
                    if reps % 4 == 0:
                        wandb.log({f"test/{f_name}": loss}, step=self.step)
                prediction.append(y_hat)
                ground_truth.append(y["final_fidelity"])

                self._step += 1
        prediction = torch.cat(prediction, dim=0)  # prediction has shape of [num_datasets, n_algos, n_fidelities]
        ground_truth = torch.cat(ground_truth, dim=0)
        return prediction, ground_truth


