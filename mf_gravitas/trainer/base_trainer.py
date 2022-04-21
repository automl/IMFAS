import torch
from tqdm import tqdm


def train(model, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001):
    losses = []

    # fixme: make optimizer a choice!
    optimizer = torch.optim.Adam(model.parameters(), lr)
    for e in tqdm(range(int(epochs))):
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

            # look if there is representation collapse:

            # calculate "attracting" forces.
            loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, model.Z_algo)

            # gradient step
            loss.backward()
            optimizer.step()
            # TODO check convergence: look if neither Z_algo nor Z_data move anymore! ( infrequently)

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
