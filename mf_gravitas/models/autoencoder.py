from typing import List

import torch
import torch.distributions as td
import torch.nn as nn

from ..util import check_diversity

class ParticleGravityAutoencoder(nn.Module):
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
        weights = [weights[0], *weights[2:], weights[1]]  # fixme: change the doc instead!
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
        self.Z_algo = nn.Parameter(td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim]))

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
        similarity_order_ind = torch.stack(
            [torch.argsort(self.cossim(a0, a1)) for a0, a1 in zip(A0, A1)]
        )
        no_repellent = int(no_comparisons * self.repellent_share)

        # find the repellent forces
        repellents = similarity_order_ind[:, :no_repellent]

        Z1_repellents = torch.stack([z1[r] for z1, r in zip(Z1_data, repellents)])

        # NOTE Producing Nans
        A1_repellents = torch.stack([a1[r] for a1, r in zip(A1, repellents)])
        mutual_weighted_dist = [
            (1 - self.cossim(a0, a1)) @ torch.linalg.norm((z0 - z1), dim=1)
            for z0, z1, a0, a1 in zip(Z0_data, Z1_repellents, A0, A1_repellents)
        ]

        data_repellent = (len(Z1_data) * len(Z1_repellents[0])) ** -1 * sum(mutual_weighted_dist)

        # find the attracting forces
        attractors = similarity_order_ind[:, no_repellent:]
        Z1_attractors = torch.stack([z1[att] for z1, att in zip(Z1_data, attractors)])
        A1_attractors = torch.stack([a1[att] for a1, att in zip(A1, attractors)])
        mutual_weighted_dist = [
            self.cossim(a0, a1) @ torch.linalg.norm((z0 - z1), dim=1)
            for z0, z1, a0, a1 in zip(Z0_data, Z1_attractors, A0, A1_attractors)
        ]
        data_attractor = (len(Z1_data) * len(Z1_attractors[0])) ** -1 * sum(mutual_weighted_dist)

        return (torch.stack([data_attractor, (-1) * data_repellent, reconstruction]) @
                self.weights[:3])

    def _loss_algorithms(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):
        # Algorithm performance "gravity" towards dataset (D0: calcualted batchwise)
        # TODO check that direction (sign) is correct!
        # compute the distance between algos and D0 (batch) dataset in embedding space
        # and weigh the distances by the algorithm's performances
        # --> pull is normalized by batch size & number of algorithms
        # Fixme: make list comprehension more pytorch style by appropriate broadcasting
        # TODO use torch.cdist for distance matrix calculation!

        '''
            Intended Psuedocode
            ----------------------------------------------------

            Z0_data -> dataset embeddings in a batch ( batch_size x latent_dim)
            Z_algo -> all the algorithm embeddings ( n_algos x latent_dim)
            A0 -> algorithm performances in a batch ( batch_size x n_algos)


            norms = []
            for z in Z0_data:
                temp = []
                for x in Z_algo:
                    temp.append(torch.norm(z - x))

                len(temp) = n_algos

                norms.append(temp)

            norms.shape == (batch_size, n_algos)


            # We want to weigh the distances
            cross = []

            for a, norm in zip(A0, norms):
                len(a) = n_algos
                len(norm) = batch_size
                cross.append(a @ norm)
        '''

        dataset_algo_distance = [
            a @ torch.linalg.norm((z - Z_algo), dim=1) for z, a in zip(Z0_data, A0)
        ]
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

        gravity = self._loss_datasets(D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo)

        return torch.stack(
            [
                gravity,
                self.weights[-1] * algo_pull,
            ]
        ).sum()

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