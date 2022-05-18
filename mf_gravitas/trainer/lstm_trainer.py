import torch
import pdb

class Trainer_Ensemble_lstm:
    def __init__(self, model, loss_fn, ranking_fn, optimizer):
        self.step = 0
        self.losses = {
            # 'ranking_loss': 0
        }

        self.model = model
        self.loss_fn = loss_fn
        self.ranking_fn = ranking_fn
        self.optimizer = optimizer

        #self.n_slices = self.model.n_fidelities

        self.loss_kwargs = {
            'ranking_fn': self.ranking_fn,
        }

    def evaluate(self, test_dataloader):
        test_lstm_losses = []

        for _, data in enumerate(test_dataloader):

            # Seperate labels into indices of slices
            D0 = data[0].to(self.model.device)

           
            labels = data[1][:,:-1,:].to(self.model.device)
            
           
            # Get Embeddings
            self.model.eval()
            with torch.no_grad():

                # Feed the lstm till penultimate fidelitie
                # calculate embedding
                shared_D0, lstm_D0 = self.model.forward(
                                    dataset_meta_features= D0,
                                    fidelities = labels
                                )

                # Get the loss for lstm output
                lstm_loss = self.loss_fn(
                        pred=lstm_D0,
                        target=data[1][:,-1,:],
                        **self.loss_kwargs
                    )


                test_lstm_losses.append(lstm_loss)
            
        self.losses[f'lstm_loss'] = torch.stack(test_lstm_losses).mean()



    def step_next(self):
        self.step += 1

    def train(self, train_dataloader):

        for _, data in enumerate(train_dataloader):

            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels
            D0 = data[0].to(self.model.device)

            labels = data[1][:,:-1,:]
            
            # calculate embedding
            shared_D0, lstm_D0 = self.model.forward(
                                    dataset_meta_features= D0,
                                    fidelities = labels
                                )

            lstm_loss = self.loss_fn(
                            pred=lstm_D0,
                            target=data[1][:,-1,:],
                            **self.loss_kwargs
                        )

            

            lstm_loss.backward()
            
            self.optimizer.step()
