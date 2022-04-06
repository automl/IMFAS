import torch
import torch.nn as nn

from gravitas.autoencoder import AE


class VAE(AE):
    # TODO allow for algo meta features
    # def __init__(
    #     self,
    #     input_dim: int = 10,
    #     hidden_dims: List[int] = [8,4],
    #     embedding_dim: int = 2,
    #     weights: List[float]=[1.0, 1.0, 1.0, 1.0],
    #     repellent_share: float =0.33,
    #     n_algos: int =20,
    #     device=None,
    # ):
    #     """
    #
    #     :param nodes: list of number of nodes from input to output
    #     :param weights: list of floats indicating the weights in the loss:
    #     reconstruction, algorithm pull towards datasets, data-similarity-attraction,
    #     data-dissimilarity-repelling.
    #     :param n_algos: number of algorithms to place in the embedding space
    #     """
    #     super().__init__()
    #     self.device = device
    #     self.weights = torch.tensor(weights).to(device)
    #     self.repellent_share = repellent_share
    #
    #     # construct the autoencoder
    #     self.embedding_dim = embedding_dim
    #     self.input_dim = input_dim
    #     self.hidden_dims = hidden_dims
    #
    #     self._build_network()
    #
    #     # initialize the algorithms in embedding space
    #     self.n_algos = n_algos
    #     self.embedding_dim = self.embedding_dim
    #     self.Z_algo = nn.Parameter(
    #         td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim])
    #     )
    #
    #     self.to(self.device)

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

        
        self.encoder = torch.nn.Sequential(*modules)

        # Mean and std_dev for the latent distribution
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], self.embedding_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1], self.embedding_dim)

        # modules.append(nn.Linear(input_dim, self.latent_dim))
        # modules.append(nn.BatchNorm1d(self.latent_dim))
        # modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.ReLU())

        

        # Make the decoder
        modules = []

        hidden_dims.reverse()
        input_dim = self.embedding_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.input_dim))
        modules.append(nn.Sigmoid()) 


        self.decoder = nn.Sequential(*modules)

    
    def reparameterize(
                    self, 
                    mu: torch.Tensor, 
                    logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return eps * std + mu
    
    def encode(self, x):
        
        # Forward pass the input through the network
        result = self.encoder(x)

        # Get the mean and standard deviation from the output 
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # TODO: Plot latent distributions



        # Sample a latent vector using the reparameterization trick
        z = self.reparameterize(mu, log_var)

        return z        

    def decode(self, x):
        return self.decoder(x)
