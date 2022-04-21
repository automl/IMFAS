import torch
from tqdm import tqdm

import numpy as np
import wandb

import torch.nn as nn



# NOTE Potentially make trainer fully abstract and have different kinds of trainers for different modules

class Trainer_Autoencoder:
    def __init__(self):
        self.step = 0
        self.losses = {
            '_loss_reconstruction': 0, 
            '_loss_datasets': 0,
            '_loss_algorithms': 0,
            'loss_gravity': 0
        }
        

    def train(self, model, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001):


        # fixme: make optimizer a choice!
        optimizer = torch.optim.Adam(model.parameters(), lr)
        for e in tqdm(range(int(epochs))):

            losses = []
            for i, data in enumerate(train_dataloader):
                (D0, A0), (D1, A1) = data

                D0 = D0.to(model.device)
                D1 = D1.to(model.device)

                A0 = A0.to(model.device)
                A1 = A1.to(model.device)
                optimizer.zero_grad()

                # calculate embedding
                D0_fwd = model.forward(D0)

                # todo not recalculate the encoding
                Z0_data = model.encode(D0)
                Z1_data = torch.stack([model.encode(d) for d in D1])

                # TODO Check for representation collapse

                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, model.Z_algo)

                # gradient step
                loss.backward()
                optimizer.step()
                losses.append(loss)
                # TODO check convergence: look if neither Z_algo nor Z_data move anymore! ( infrequently)

            # log losses
            self.losses[loss_fn.__name__] = losses[-1]
            wandb.log(
                self.losses,
                commit=False,
                step=self.step
            )


            self.step += 1

            # TODO wandb logging

            
            
            
            # validation every e epochs
            test_timer = 10
            test_losses = []
            # if e % test_timer == 0:
            #     # look at the gradient step's effects on validation data
            #     D_test = train_dataloader.dataset.datasets_meta_features
            #     D_test = D_test.to(model.device)
            #     Z_data = model.encode(D_test)
            #
            #     tracking.append((model.Z_algo.data.clone(), Z_data))

            # TODO validation procedure

    def _freeze(self, listoflayers, unfreeze=True):
        """freeze the parameters of the list of layers """
        for l in listoflayers:
            for p in l.parameters():
                p.requires_grad = unfreeze


    def freeze_train(self, model, epochs, *args, **kwargs):
        """
        unfreeze from the outer to the inner of the autoencoder
        :param epochs: total epochs that are to be trained
        :param args: args passed to _train()
        """
        # find those layers that need to be frozen
        enc = [l for l in model.encoder if isinstance(l, nn.Linear)]
        dec = [l for l in reversed(model.decoder) if isinstance(l, nn.Linear)]

        enc_bn = [l for l in model.encoder if isinstance(l, nn.BatchNorm1d)]
        dec_bn = [l for l in reversed(model.encoder) if isinstance(l, nn.BatchNorm1d)]

        # linear intervals for unfreezing
        no_freeze_steps = len(enc)

        # freeze all intermediate layers
        self._freeze([*enc, *dec, *enc_bn, *dec_bn], unfreeze=False)

        # scheduler for the layers that are iteratively unfrozen
        unfreezer = zip(enc, dec, enc_bn, dec_bn)

        for i in range(no_freeze_steps):
            # determine which layers are melted
            en, de, ebn, dbn = unfreezer.__next__()
            self._freeze([en, de, ebn, dbn], unfreeze=True)
        self.train(model, epochs=epochs // no_freeze_steps, *args, **kwargs)
