import torch


# FIXME: does not track using wandb!

class Trainer_Ensemble:
    def __init__(self, model, loss_fn, ranking_fn, optimizer):
        self.step = 0
        self.losses = {
            # 'ranking_loss': 0
        }

        self.model = model
        self.loss_fn = loss_fn
        self.ranking_fn = ranking_fn
        self.optimizer = optimizer

        self.n_slices = self.model.n_multiheads + 1

        self.loss_kwargs = {
            'ranking_fn': self.ranking_fn,
        }

    def evaluate(self, test_dataloader):
        test_final_losses = []
        test_multi_head_losses = []

        for _, data in enumerate(test_dataloader):

            # Dataset meta features and final  slice labels
            D0 = data[0].to(self.model.device)

            labels = []
            for j in range(self.n_slices):
                labels.append(data[1][0, j].reshape(1, -1).to(self.model.device))

            # calculate embedding
            self.model.eval()
            with torch.no_grad():
                shared_D0, multi_head_D0, final_D0 = self.model.forward(D0)

                # Get the multihead losses
                multi_head_losses = []
                for j in range(self.n_slices - 1):
                    # Calculate the loss on multi heads
                    multi_head_loss = self.loss_fn(
                        pred=multi_head_D0[j],
                        target=labels[j],
                        **self.loss_kwargs
                    )
                    multi_head_losses.append(multi_head_loss)

                # Calculate the loss on final layer
                final_loss = self.loss_fn(
                    pred=final_D0,
                    target=labels[-1],
                    **self.loss_kwargs
                )

                test_final_losses.append(final_loss)
                [test_multi_head_losses.append(l) for l in multi_head_losses]
            return multi_head_losses

        for j in range(self.n_slices - 1):
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
                labels.append(data[1][0, j].reshape(1, -1).to(self.model.device))

            # calculate embedding
            shared_D0, multi_head_D0, final_D0 = self.model.forward(D0)

            multi_head_losses = []
            for j in range(self.n_slices - 1):
                # Calculate the loss on multi heads
                multi_head_loss = self.loss_fn(
                    pred=multi_head_D0[j],
                    target=labels[j],
                    **self.loss_kwargs
                )
                multi_head_losses.append(multi_head_loss)

            # Calculate the loss on final head
            multi_head_sum = sum(multi_head_losses)

            # calculate the loss on final layer
            final_loss = self.loss_fn(
                pred=final_D0,
                target=labels[-1],
                **self.loss_kwargs
            )

            joint_loss = multi_head_sum + final_loss

            joint_loss.backward()
            self.optimizer.step()
