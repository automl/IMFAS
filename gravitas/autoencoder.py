from typing import List

import torch
import torch.distributions as td
import torch.nn as nn
from tqdm import tqdm

from gravitas.base_encoder import BaseEncoder


class AE(BaseEncoder):
    # TODO allow for algo meta features
    def __init__(
            self,
            input_dim: int = 10,
            hidden_dims: List[int] = [8, 4],
            embedding_dim: int = 2,
            weights=[1.0, 1.0, 1.0, 1.0],
            repellent_share=0.33,
            n_algos=20,
            device=None,
    ):
        """

        :param nodes: list of number of nodes from input to output
        :param weights: list of floats indicating the weights in the loss:
        reconstruction, algorithm pull towards datasets, data-similarity-attraction,
        data-dissimilarity-repelling.
        :param n_algos: number of algorithms to place in the embedding space
        """
        super().__init__()
        self.device = device
        weights = [weights[0], *weights[2:], weights[1]] # fixme: change the doc instead!
        self.weights = torch.tensor(weights).to(device)
        self.repellent_share = repellent_share

        # construct the autoencoder
        self.latent_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        # initialize the algorithms in embedding space
        self.n_algos = n_algos
        self.embedding_dim = self.latent_dim
        self.Z_algo = nn.Parameter(
            td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim])
        )

        self._build_network()
        self.to(self.device)

    def _build_network(self) -> None:
        """
        Builds the encoder and decoder networks
        """
        # Make the Encoder
        modules = []

        hidden_dims = self.hidden_dims
        input_dim = self.input_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.latent_dim))
        modules.append(nn.BatchNorm1d(self.latent_dim))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.ReLU())

        self.encoder = torch.nn.Sequential(*modules)

        # Make the decoder
        modules = []

        hidden_dims.reverse()
        input_dim = self.latent_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.input_dim))
        modules.append(nn.Sigmoid())  # input_dim, self.input_dim

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, D):
        """
        Forward path through the meta-feature autoencoder
        :param D: input tensor
        :return: tuple: output tensor
        """
        return self.decode(self.encode(D))

    def _loss_reconstruction(self, D0, D0_fwd, *args, **kwargs):
        # reconstruction loss (Autoencoder loss)
        # its purpose is to avoid simple single point solution with catastrophic
        # information loss - in the absence of a repelling force.
        return torch.nn.functional.mse_loss(D0, D0_fwd)

    def _loss_datasets(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):

        reconstruction = self._loss_reconstruction(D0, D0_fwd)
        # batch similarity order + no_repellents
        no_comparisons = A1.shape[1]
        similarity_order_ind = torch.stack([torch.argsort(self.cossim(a0, a1)) for a0, a1 in zip(A0, A1)])
        no_repellent = int(no_comparisons * self.repellent_share)

        # find the repellent forces
        repellents = similarity_order_ind[:, :no_repellent]
        Z1_repellents = torch.stack([z1[r] for z1, r in zip(Z1_data, repellents)])
        A1_repellents = torch.stack([a1[r] for a1, r in zip(A1, repellents)])
        mutual_weighted_dist = [(1 - self.cossim(a0, a1)) @ torch.linalg.norm((z0 - z1), dim=1)
                                for z0, z1, a0, a1 in zip(Z0_data, Z1_repellents, A0, A1_repellents)]
        data_repellent = (len(Z1_data) * len(Z1_repellents[0])) ** -1 * sum(mutual_weighted_dist)

        # find the attracting forces
        attractors = similarity_order_ind[:, no_repellent:]
        Z1_attractors = torch.stack([z1[att] for z1, att in zip(Z1_data, attractors)])
        A1_attractors = torch.stack([a1[att] for a1, att in zip(A1, attractors)])
        mutual_weighted_dist = [self.cossim(a0, a1) @ torch.linalg.norm((z0 - z1), dim=1)
                                for z0, z1, a0, a1 in zip(Z0_data, Z1_attractors, A0, A1_attractors)]
        data_attractor = (len(Z1_data) * len(Z1_attractors[0])) ** -1 * sum(mutual_weighted_dist)

        return torch.stack([data_attractor, (-1) * data_repellent, reconstruction]) @ self.weights[:3]

    def _loss_algorithms(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):
        # Algorithm performance "gravity" towards dataset (D0: calcualted batchwise)
        # TODO check that direction (sign) is correct!
        # compute the distance between algos and D0 (batch) dataset in embedding space
        # and weigh the distances by the algorithm's performances
        # --> pull is normalized by batch size & number of algorithms
        # Fixme: make list comprehension more pytorch style by apropriate broadcasting
        # TODO use torch.cdist for distance matrix calculation!
        dataset_algo_distance = [a @ torch.linalg.norm((z - Z_algo), dim=1) for z, a in zip(Z0_data, A0)]
        return (len(Z_algo) * len(Z0_data)) ** -1 * sum(dataset_algo_distance)

    def loss_gravity(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):
        """
        Creates a pairwise (dataset-wise) loss that
        a) enforces a reconstruction of the datasets meta features (i.e. we
        have a meaningful embedding)
        b) ensure, that algorithms that perform good on datasets are drawn towards
        those datasets in embedding space.
        c) pull together datasets, if similar algorithms performed well on them.

        # Consider: use of squared/linear/learned exponential (based on full
        # prediction: top_k selection) algo performance for weighing?
        # Consider: that the 90% performance should also be part of the loss
        # this might allow to get a ranking immediately from the distances in
        # embedding space!

        :param D0: Dataset 0 meta features
        :param D0_fwd: autoencoder reconstruction of Dataset 0 meta features
        :param Z0_data: embedding of dataset 0 meta features
        :param Z1_data: embedding of dataset 1 meta features
        :param A0: vector of algorithm performances on dataset 0
        :param A1: vector of algorithm performances on dataset 1
        :param Z_algo: algorithm embedding vector of same dim as Z_data


        :return: scalar.
        """

        algo_pull = self._loss_algorithms(D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo)

        gravity = self._loss_datasets(
            D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo)

        return torch.stack([gravity, self.weights[-1] * algo_pull,])

    def train_gravity(self, train_dataloader, test_dataloader, epochs, lr=0.001):
        """
        Two step training:
        1) Pretraining using reconstruction loss
        2) Training using gravity loss (reconstruction + attraction + repellent +
        :param train_dataloader:
        :param test_dataloader:
        :param epochs: list of int (len 2): epochs for step 1) and 2) respectively
        :param lr:
        :return:
        """
        name = self.__class__.__name__
        print(f'\nPretraining {name} with reconstruction loss: ')
        self._train(self._loss_reconstruction, train_dataloader, test_dataloader, epochs[0])

        print(f'\nTraining {name} with gravity loss:')
        return self._train(self.loss_gravity, train_dataloader, test_dataloader, epochs[1], lr=lr)

    def train_schedule(self, train_dataloader, test_dataloader, epochs=[100, 100, 100], lr=0.001):
        # Consider Marius idea to first find a reasonable data representation
        #  and only than train with the algorithms

        # pretrain
        name = self.__class__.__name__
        print(f'\nPretraining {name} with reconstruction loss:')
        self._train(self._loss_reconstruction, train_dataloader, test_dataloader, epochs[0], lr)

        # train datasets
        print(f'\nTraining {name} with dataset loss:')
        self._train(self._loss_datasets, train_dataloader, test_dataloader, epochs[1], lr)

        # train algorithms
        print(f'\nTraining {name} with algorithm:')
        return self._train(self._loss_algorithms, train_dataloader, test_dataloader, epochs[2], lr)


    def _train(self, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001):
        losses = []
        test_losses = []

        tracking = []
        optimizer = torch.optim.Adam(self.parameters(), lr)
        for e in tqdm(range(epochs)):
            for i, data in enumerate(train_dataloader):
                D0, D1, A0, A1 = data

                D0 = D0.to(self.device)
                D1 = D1.to(self.device)
                A0 = A0.to(self.device)
                A1 = A1.to(self.device)
                optimizer.zero_grad()

                # calculate embedding
                D0_fwd = self.forward(D0)

                # todo not recalculate the encoding
                Z0_data = self.encode(D0)
                Z1_data = torch.stack([self.encode(d) for d in D1])

                # look if there is representation collapse:

                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, self.Z_algo)

                # gradient step
                loss.backward()
                optimizer.step()
                # TODO check convergence: look if neither Z_algo nor Z_data move anymore! ( infrequently)

            losses.append(loss)

            # validation every e epochs
            test_timer = 10
            test_losses = []
            # if e % test_timer == 0:
            #     # look at the gradient step's effects on validation data
            #     D_test = train_dataloader.dataset.datasets_meta_features
            #     D_test = D_test.to(self.device)
            #     Z_data = self.encode(D_test)
            #
            #     tracking.append((self.Z_algo.data.clone(), Z_data))

                # TODO validation procedure

        return tracking, losses, test_losses

    def predict_algorithms(self, D, topk):
        """
        Find the topk performing algorithm candidates.

        :param D: meta features of dataset D
        :param topk: number of candidate algorithms to return.
        :return: set of indicies representing likely good performing algorithms based on their
        distance in embedding space.
        """
        # embed dataset.
        self.eval()

        with torch.no_grad():
            Z_data = self.encode(D)

            # find k-nearest algorithms.
            # sort by distance in embedding space.
            dist_mat = torch.cdist(Z_data, self.Z_algo)
            _, top_algo = torch.topk(dist_mat, largest=False, k=topk)  # find minimum distance

        self.train()

        return top_algo
