import torch


class Trainer_Ensemble_lstm:
    def __init__(self, model, loss_fn, ranking_fn, optimizer, test_lim=5):
        self.step = 0
        self.losses = {
            # 'ranking_loss': 0
        }

        self.model = model
        self.loss_fn = loss_fn
        self.ranking_fn = ranking_fn
        self.optimizer = optimizer
        self.test_lim = test_lim
        self.readout = torch.nn.Linear(model.shared_hidden_dims[-1], model.algo_dim)

        # self.n_slices = self.model.n_fidelities

        self.loss_kwargs = {
            "ranking_fn": self.ranking_fn,
        }

    def evaluate(self, test_dataloader):
        test_lstm_losses = []
        test_shared_losses = []

        for _, data in enumerate(test_dataloader):
            # Seperate labels into indices of slices
            D0 = data[0].to(self.model.device)

            # Only feed limited fidelities for test
            labels = data[1][:, : self.test_lim, :].to(self.model.device)

            # Get Embeddings
            self.model.eval()
            with torch.no_grad():
                # Feed the lstm till penultimate fidelitie
                # calculate embedding
                shared_D0, lstm_D0 = self.model.forward(dataset_meta_features=D0, fidelities=labels)

                # Get the loss for lstm output
                lstm_loss = self.loss_fn(pred=lstm_D0, target=data[1][:, -1, :], **self.loss_kwargs)

                self.readout.load_state_dict(self.model.lstm_network.readout.state_dict())
                D0_rank = self.readout.forward(shared_D0.detach())

                # Get the loss for lstm output
                shared_loss = self.loss_fn(pred=D0_rank.detach(), target=data[1][:, -1, :], **self.loss_kwargs)

                test_lstm_losses.append(lstm_loss)
                test_shared_losses.append(shared_loss)

        self.losses[f"lstm_loss"] = torch.stack(test_lstm_losses).mean()
        self.losses[f"shared_loss"] = torch.stack(test_shared_losses).mean()

    def step_next(self):
        self.step += 1

    def train(self, train_dataloader):

        for _, data in enumerate(train_dataloader):
            self.optimizer.zero_grad()
            # Dataset meta features and final  slice labels
            D0 = data[0].to(self.model.device)

            labels = data[1][:, :-1, :]

            # calculate embedding
            shared_D0, lstm_D0 = self.model.forward(dataset_meta_features=D0, fidelities=labels)

            lstm_loss = self.loss_fn(pred=lstm_D0, target=data[1][:, -1, :], **self.loss_kwargs)

            lstm_loss.backward()

            self.optimizer.step()
