"""
As a proof of concept for m-shot, we need to show that we can find configurations
in a zero_shot manner we could reach. To do so, we train our manifold & the algo
instantiations and densely sample the latent space and compute the ranking for
each of these points (using the model's rank prediction module)
"""

import torch

from mf_gravitas.data import DatasetMetaFeatures


class ZeroShotOptimalDistance:
    def __init__(self, model, ranking_loss):
        self.model = model
        self.ranking_loss = ranking_loss
        self.batch = 20  # TODO unfix here!

    def forward(self, dataset_meta_features: DatasetMetaFeatures, steps):
        """
        Defining the evaluation protocol. Since ZeroShotOptimalDistance
        is a static method, forward does not depend on some running meth
        """

        self.dataset_meta_features = dataset_meta_features
        self.loader = torch.utils.data.DataLoader(
            self.dataset_meta_features,
            batch_size=self.batch,  # Since we are only encode on the model we can crank this up
            shuffle=False,  # important to see where it is going wrong
            num_workers=2
        )
        self.n_algos = len(self.loader.dataset.algo_names)
        Y_hat, modelrankings, A0_full = self.encode_loader()
        cuboid_dims = self._get_bounds(Y_hat)
        cuboid_rankings = self.get_cuboid_rankings(cuboid_dims, steps)
        self._compare_rankings(cuboid_rankings, modelrankings, A0_full)

    def encode_loader(self):
        """
        Pre-allocate & gather all the embeddings for the dataset at hand
        """

        # TODO Use this process to calculate the E-MSE as separate class!
        n = len(self.dataset_meta_features)

        Y_hat = torch.zeros((n, self.model.latent_dim))
        rankings = torch.zeros((n, self.n_algos))
        A0_full = torch.zeros((n, self.n_algos))
        for i, data in enumerate(self.loader):
            (D0, A0), _ = data

            # D0 = D0.to(self.device) , .....
            D0.to(self.model.device)
            A0_full[i:(i + self.batch)] = A0

            # predict the rankings of that point
            Y_hat[i:(i + self.batch)] = self.model.encode(D0)

            # calculate the model's rankings
            # consider reusing computation here!
            rankings[i:(i + self.batch)] = self.model.predict_algorithms(D0, topk=self.n_algos)

        return Y_hat, rankings, A0_full

    def _get_bounds(self, Y_hat):
        """
        To calculate the distances exhaustively, we need to find the cuboid
        bounds we would like to sample in
        """
        # NOTE: could we use the algo vectors as bounds?

        # find the bounds of the cuboid
        # (columnwise min & max)
        min_bounds = torch.min(Y_hat, dim=0).values.tolist()
        max_bounds = torch.max(Y_hat, dim=0).values.tolist()

        # splice them such that we have the zip of them
        cuboid_dims = [dims for dims in zip(min_bounds, max_bounds)]
        return cuboid_dims

    def get_cuboid_rankings(self, cuboid_dims, steps):
        self.model.eval()
        with torch.no_grad():
            # grid the alternatives
            support = [torch.linspace(l, h, steps) for l, h in cuboid_dims]
            latent_coords = torch.meshgrid(support)

            # TODO check carefully if stacking is done as expected
            cuboid = torch.stack(latent_coords, dim=-1).reshape((-1, 2))

            # TODO to device?

            #  calc the rankings for each grid point (relative to the model's algorithm vector
            #  instantiations
            dist_mat = torch.cdist(cuboid, self.model.Z_algo)
            _, rankings = torch.topk(dist_mat, largest=False, k=self.n_algos)

        return rankings

    def _compare_rankings(self, cuboid_rankings, rankings, A0):
        """
        Try to find positions in the latent space that produce an ordinal better ranking

        :param cuboid_rankings: tensor of all ranking vectors for each grid point
        :param rankings: tensor for all ranking vectors for all dataset points (from loader)
        :param A0: tensor: algorithm performances.

        :return:
        """

        cuboid_rankings

        rankings

        _, true_rankings = torch.topk(A0, largest=True, k=self.n_algos)

        # now compare model rankings with true rankings so that we see how far
        # we have gotten with training

        # afterwards compute for the cuboid-groundtruth loss for each datasetpoint

        # check if there is at lest one dataset point in the cuboid which is closer to the truth
        return self.ranking_loss(truth, predicted)
