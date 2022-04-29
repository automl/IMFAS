import torch
from tqdm import tqdm

import numpy as np
import wandb

import torch.nn as nn
from ..util import measure_embedding_diversity

import pdb
import torchsort
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

                # # TODO Check for representation collapse
               
                # pdb.set_trace()


                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, model.Z_algo)

                # gradient step
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())
                # TODO check convergence: look if neither Z_algo nor Z_data move anymore! ( infrequently)

            d_div, z_div = measure_embedding_diversity(
                                        model = model, 
                                        data = train_dataloader.dataset.meta_dataset.transformed_df[train_dataloader.dataset.splitindex]
                                    )

            
            wandb.log(
                {
                    'data_div': d_div,
                    'algo_div': z_div,
                },
                commit=False,
                step=self.step
            )

           

            
            test_losses = []
            for i, data in enumerate(test_dataloader):
                (D0, A0), (D1, A1) = data

                D0 = D0.to(model.device)
                D1 = D1.to(model.device)

                A0 = A0.to(model.device)
                A1 = A1.to(model.device)
                # calculate embedding
                D0_fwd = model.forward(D0)

                # todo not recalculate the encoding
                Z0_data = model.encode(D0)
                Z1_data = torch.stack([model.encode(d) for d in D1])

                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, model.Z_algo)
                test_losses.append(loss.detach().item())

            # self.losses[loss_fn.__name__] = test_losses[-1]
            # wandb.log(
            #     self.losses,
            #     commit=False,
            #     step=self.step
            # )

            # log losses
            self.losses[loss_fn.__name__] = np.mean(test_losses)
            wandb.log(
                self.losses,
                commit=False,
                step=self.step
            )
                

            self.step += 1


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


class Trainer_Rank:
    def __init__(self):
        self.step = 0
        self.losses = {
            #'ranking_loss': 0
        }
        

    def train(self, model, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001, log_wandb=True, slice_index=-1):

        
        optimizer = torch.optim.Adam(model.parameters(), lr)
        for e in tqdm(range(int(epochs))):

            losses = []
            for i, data in enumerate(train_dataloader):
                
                # Dataset meta features and final  slice labels
                D0 = data[0].to(model.device)
                labels = data[1][0,slice_index].reshape(1,-1).to(model.device)

                # calculate embedding
                D0_fwd = model.forward(D0)

                # Calculate the loss
                loss = loss_fn(
                    pred = D0_fwd,
                    target = labels,
                )


                # gradient step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().item())

            
            test_losses = []
            for i, data in enumerate(test_dataloader):
                
                 # Dataset meta features and final  slice labels
                D0 = data[0].to(model.device)
                labels = data[1][0,slice_index].reshape(1,-1).to(model.device)

                # calculate embedding
                D0_fwd = model.forward(D0)

                # Calculate the loss
                loss = loss_fn(
                    pred = D0_fwd,
                    target = labels,
                )
                
                test_losses.append(loss.detach().item())

            # log losses
            self.losses[f'{slice_index}/ranking_loss'] = np.mean(test_losses)


            if log_wandb:
                wandb.log(
                    self.losses,
                    commit=False,
                    step=self.step
                )
                    

            self.step += 1        

        return {
            'loss' : self.losses[f'{slice_index}/ranking_loss'],
            'step' : self.step - 1
        }


class Trainer_Ensemble:
    def __init__(self, model, loss_fn, ranking_fn, optimizer):
        self.step = 0
        self.losses = {
            #'ranking_loss': 0
        }

        self.model = model
        self.loss_fn = loss_fn
        self.ranking_fn = ranking_fn
        self.optimizer = optimizer

        self.n_slices = self.model.n_fidelities + 1 
        
        self.loss_kwargs = {
            'ranking_fn' : self.ranking_fn,
        }
    
    
    def evaluate(self, test_dataloader):
        test_final_losses = []
        test_multi_head_losses = []
        
        for _, data in enumerate(test_dataloader):
                
            # Dataset meta features and final  slice labels
            D0 = data[0].to(self.model.device)

            labels = []
            for j in range(self.n_slices):
                labels.append(data[1][0,j].reshape(1,-1).to(self.model.device))

            # calculate embedding    
            self.model.eval()
            with torch.no_grad():
                shared_D0, multi_head_D0, final_D0 = self.model.forward(D0)

                # Get the multihead losses
                multi_head_losses = []
                for j in range(self.n_slices-1):
                    
                    # Calculate the loss on multi heads
                    multi_head_loss = self.loss_fn(
                        pred = multi_head_D0[j],
                        target = labels[j],
                        **self.loss_kwargs
                    )

                    multi_head_losses.append(multi_head_loss)
                
                # Calculate the loss on final layer
                final_loss = self.loss_fn(
                    pred = final_D0,
                    target = labels[-1],
                    **self.loss_kwargs
                )

                test_final_losses.append(final_loss)
                [test_multi_head_losses.append(l) for l in multi_head_losses]

        for j in range(self.n_slices-1):
                self.losses[f'{j}/multihead_loss'] = test_multi_head_losses[j]
            
        self.losses[f'final_ranking_loss'] = torch.stack(test_final_losses).mean()


    def step_next(self):
        self.step += 1

    
    def train(self, train_dataloader):

        for _, data in enumerate(train_dataloader):
            
            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels
            D0 = data[0].to(self.model.device)

            labels = []
            for j in range(self.n_slices):
                labels.append(data[1][0,j].reshape(1,-1).to(self.model.device))
            
            # calculate embedding
            shared_D0, multi_head_D0, final_D0 = self.model.forward(D0)

            multi_head_losses = []
            for j in range(self.n_slices-1):
                
                # Calculate the loss on multi heads
                multi_head_loss = self.loss_fn(
                    pred = multi_head_D0[j],
                    target = labels[j],
                    **self.loss_kwargs
                )
                multi_head_losses.append(multi_head_loss)
            
            # Calculate the loss on final head
            multi_head_sum = sum(multi_head_losses)
        
            #calculate the loss on final layer
            final_loss = self.loss_fn(
                pred = final_D0,
                target = labels[-1],
                **self.loss_kwargs
            )

            joint_loss = multi_head_sum + final_loss

            joint_loss.backward()
            self.optimizer.step()

                

    