import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import *
import torch.nn.functional as F
import pdb
from torchpq.clustering import KMeans
from torch.distributions import Categorical, Normal, MixtureSameFamily


class WAE1D(nn.Module):
    def __init__(
        self,
        prior,
        in_dim: Optional[int],
        x_dim: Optional[int] = None,
        out_shape: Optional[Tuple[int,int,int]] = None,
        h_dim: int = 256,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
        embedding_dim: int = 128,
        likelihood: str = None,
        dropout: float = 0.0,
        out_activation: Optional[str] = None,
        auto_init: int = 1,
        **kwargs,
    ):
        super().__init__()
        assert (x_dim is not None) ^ (out_shape is not None), \
            "Specify exactly one of x_dim or out_shape."

        self.prior = prior
        # Encoder: outputs (B, embedding_dim)
        self.encoder = Encoder(
            in_dim=in_dim, h_dim=h_dim, n_layers=n_layers_enc,
            out_dim=embedding_dim, dropout=dropout
        )

        self.likelihood = likelihood
        ### Gaussian decoder
        if likelihood == None:
            self.likelihood = "gaussian"
            self.decoder = Decoder(
                z_dim=embedding_dim,
                h_dim=h_dim,
                n_layers=n_layers_dec,
                x_dim=x_dim,
                out_shape=out_shape,
                out_activation=out_activation,
                dropout=dropout,
            )
        elif likelihood == "nb":
            self.decoder = NBDecoder(
                z_dim=embedding_dim,
                h_dim=h_dim,
                n_layers=n_layers_dec,
                x_dim=x_dim,
                dropout=dropout,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported likelihood: {likelihood}")
        
        assert auto_init in [0,1]
        if auto_init:
            self.register_buffer('init', torch.zeros(1, dtype=torch.bool))
        else:
            self.register_buffer('init', torch.ones(1, dtype=torch.bool))
    
    def forward(self, x: torch.Tensor, weight = 10.0):
        # Encode
        #pdb.set_trace()
        #x_log1p = torch.log1p(x)
        #z = self.encoder(x_log1p)
        bs, *shape = x.shape

        if self.likelihood == "gaussian":
            z = self.encoder(x)
            x_recon = self.decoder(x, z)
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / bs
        elif self.likelihood == "nb":
            x_log1p = torch.log1p(x)
            z = self.encoder(x_log1p)
            mu,theta = self.decoder(x, z)
            recon_loss = nb_nll_from_mu_theta(x, mu, theta)
        
        if self.training and not self.init[0]:
            self._init_prior(z)
        
        wd = self.wasserstein_distance_1d(z, self.prior)
        w2 = weight * wd
        loss = recon_loss + w2

        metrics = {}
        metrics['recon_loss'] = recon_loss.detach()
        metrics['wasserstein_distance'] = wd.detach()
        metrics["loss"] = loss.detach()
        return loss, metrics
    
    def wasserstein_distance_1d(self,encoded_samples, distribution):
        bs = encoded_samples.size(0)
        #pdb.set_trace()
        z_prior = distribution.sample((bs,1)).to(encoded_samples.device)

        wasserstein_distance = F.mse_loss(
            torch.sort(encoded_samples, dim=0)[0],
            torch.sort(z_prior, dim=0)[0],
            reduction='sum'
        ) / bs
        return wasserstein_distance
    
    def _init_prior(self, z: torch.Tensor):
        #pdb.set_trace()
        n_clusters = self.prior.mixture_distribution.logits.shape[-1]
        ### use k-means to init cluster centers
        kmeans = KMeans(n_clusters=n_clusters, distance='euclidean', init_mode="kmeans++")
        _ = kmeans.fit(z.detach().T.contiguous())
        centroids = kmeans.centroids.T.contiguous().reshape(-1).to(device=z.device) *2
        var_init = torch.var(z, dim=0).mean().clone().detach().to(device=z.device)/ n_clusters
        stds = torch.sqrt(var_init).clamp_min(1e-6) * torch.ones_like(centroids)
        mix_probs = torch.full((n_clusters,), 1.0 / n_clusters, device=z.device)
        mix = Categorical(mix_probs)
        comp = Normal(centroids, stds)               # batch of K Normals
        gmm = MixtureSameFamily(mix, comp)
        self.prior = gmm
        #self.prior = self.prior.to(z.device)
        self.init[0] = True