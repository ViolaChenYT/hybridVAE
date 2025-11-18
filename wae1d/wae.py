import torch
import torch.nn as nn
import numpy as np
from encoder import Encoder
from decoder import *
import torch.nn.functional as F
import pdb
from torchpq.clustering import KMeans
from torch.distributions import Categorical, Normal, MixtureSameFamily, kl_divergence
from torch.utils.data import Dataset, DataLoader

class DistanceDataset(Dataset):
    def __init__(self, X_np, D_np):
        # X_np: (N, m), D_np: (N, N)
        self.X = torch.from_numpy(X_np).float()
        self.D = torch.from_numpy(D_np).float()
        self.N = self.X.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # return x_i and its global index i
        return self.X[idx], idx


class WAE1D(nn.Module):
    def __init__(
        self,
        prior,
        device,
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
        update_prior_mix: int = 0,
        **kwargs,
    ):
        super().__init__()
        assert (x_dim is not None) ^ (out_shape is not None), \
            "Specify exactly one of x_dim or out_shape."

        self.prior = prior
        self.device = device
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
        assert update_prior_mix in [0,1]
        self.update_prior_mix = update_prior_mix
        if auto_init:
            self.register_buffer('init', torch.zeros(1, dtype=torch.bool))
        else:
            self.register_buffer('init', torch.ones(1, dtype=torch.bool))
    
    def forward(self, x: torch.Tensor, weight = 10.0, **kwargs):
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
        
        wd = self.wasserstein_distance_1d(z, **kwargs)
        w2 = weight * wd
        loss = recon_loss + w2

        metrics = {}
        metrics['recon_loss'] = recon_loss.detach()
        metrics['wasserstein_distance'] = wd.detach()
        metrics["loss"] = loss.detach()
        return loss, metrics
    
    def _sample_gmm_with_gumbel_softmax(self, batch_size: int, tau: float = 0.5):
        #pdb.set_trace()
        comp = self.prior.component_distribution
        n_clusters = comp.loc.numel()
        loc = comp.loc.to(self.device)
        scale = comp.scale.to(self.device) 
        logits = self.prior.mixture_distribution.logits.to(self.device)   # [K]
        logits_expanded = logits.unsqueeze(0).expand(batch_size, -1)  # [B, K]

        # y: [B, K], relaxed one-hot over components
        # hard=False keeps it fully differentiable
        y = F.gumbel_softmax(logits_expanded, tau=tau, hard=False, dim=-1)

        # ----- reparameterized Gaussian components -----
        eps = torch.randn(batch_size, n_clusters, device=self.device)   # [B, K]
        z_components = loc.unsqueeze(0) + scale.unsqueeze(0) * eps  # [B, K]

        # mixture sample z = sum_k y_k * z_k
        z = (y * z_components).sum(dim=-1, keepdim=True)  # [B, 1]
        return z

    def wasserstein_distance_1d(self,encoded_samples, **kwargs):
        bs = encoded_samples.size(0)
        pdb.set_trace()
        if self.update_prior_mix:
            z_prior = self._sample_gmm_with_gumbel_softmax(bs, **kwargs)
        else:
            z_prior = self.prior.sample((bs,1)).to(encoded_samples.device)

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


class GMMWAE1D(nn.Module):
    def __init__(
        self,
        #prior,
        means,
        stds,
        n_clusters,
        device,
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
        update_prior_mix: int = 0,
        **kwargs,
    ):
        super().__init__()
        assert (x_dim is not None) ^ (out_shape is not None), \
            "Specify exactly one of x_dim or out_shape."

        #self.prior = prior
        self.device = device
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
        assert update_prior_mix in [0,1]
        self.update_prior_mix = update_prior_mix
        self.n_clusters = n_clusters
        self.means = means.to(device)
        self.stds = stds.to(device)
        if not update_prior_mix:
            mix_probs = torch.tensor([1/n_clusters for _ in range(n_clusters)])
            mix = Categorical(mix_probs)
            comp = Normal(means, stds)
            self.prior = MixtureSameFamily(mix, comp)
        else:
            self.mix_logits = nn.Parameter(torch.zeros(n_clusters))
        if auto_init:
            self.register_buffer('init', torch.zeros(1, dtype=torch.bool))
        else:
            self.register_buffer('init', torch.ones(1, dtype=torch.bool))
        self.dist_scale = nn.Parameter(torch.tensor(1.0))  # learnable scalar
    
    def forward(self, x: torch.Tensor, dist: torch.Tensor, lambda1 = 10, lambda2 = 1, weight = 10.0, beta = 1, threshold=10000, **kwargs):
        # Encode
        #pdb.set_trace()
        #x_log1p = torch.log1p(x)
        #z = self.encoder(x_log1p)
        bs, *shape = x.shape
        assert bs == dist.size(0)

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
        
        wd = self.wasserstein_distance_1d(z, **kwargs)
        w2 = weight * wd
        loss = recon_loss + w2

        metrics = {}
        metrics['recon_loss'] = recon_loss.detach()
        metrics['wasserstein_distance'] = wd.detach()

        if self.update_prior_mix:
            mix_probs = F.softmax(self.mix_logits, dim=0)
            uniform_probs = torch.full_like(mix_probs, 1.0 / self.n_clusters)
            cat_p = Categorical(probs=mix_probs)
            cat_u = Categorical(probs=uniform_probs)
            kl_reg = kl_divergence(cat_p, cat_u)
            loss += weight * beta * kl_reg
            metrics['kl_reg'] = kl_reg.detach()
        
        if lambda1>0:
            # pairwise latent distances
            z_diff = z.unsqueeze(0) - z.unsqueeze(1)
            dist_latent = z_diff.norm(p=2, dim=-1)
            # weight exp(-λ2 D_ij)
            geom_weight = torch.exp(-lambda2 * dist)
            # mask for distances within threshold
            within_threshold = (dist <= threshold)  # [bs, bs], bool
            # zero out weights where dist > th
            geom_weight = geom_weight * within_threshold
            diff2 = (dist_latent - self.dist_scale * dist)**2
            # use only i < j (upper triangle, no diagonal)
            mask = torch.triu(torch.ones(bs, bs, device=self.device), diagonal=1).bool()
            # only keep pairs that are both upper-triangular and within threshold
            mask = mask & within_threshold
            num_pairs = mask.sum().clamp(min=1)
            pairwise_loss = (geom_weight * diff2 * mask).sum() / num_pairs
            loss += lambda1 * pairwise_loss
            metrics['pairwise_loss'] = pairwise_loss.detach()
        
        metrics["loss"] = loss.detach()
        return loss, metrics
    
    def _sample_gmm_with_gumbel_softmax(self, batch_size: int, tau: float = 0.5):
        #pdb.set_trace()
        #comp = self.prior.component_distribution
        #n_clusters = comp.loc.numel()
        #loc = comp.loc.to(self.device)
        #scale = comp.scale.to(self.device) 
        #logits = self.prior.mixture_distribution.logits.to(self.device)   # [K]
        logits_expanded = self.mix_logits.unsqueeze(0).expand(batch_size, -1)  # [B, K]

        # y: [B, K], relaxed one-hot over components
        # hard=False keeps it fully differentiable
        y = F.gumbel_softmax(logits_expanded, tau=tau, hard=False, dim=-1)

        # ----- reparameterized Gaussian components -----
        eps = torch.randn(batch_size, self.n_clusters, device=self.device)   # [B, K]
        z_components = self.means.unsqueeze(0) + self.stds.unsqueeze(0) * eps  # [B, K]

        # mixture sample z = sum_k y_k * z_k
        z = (y * z_components).sum(dim=-1, keepdim=True)  # [B, 1]
        return z

    def wasserstein_distance_1d(self,encoded_samples, **kwargs):
        bs = encoded_samples.size(0)
        #pdb.set_trace()
        if self.update_prior_mix:
            z_prior = self._sample_gmm_with_gumbel_softmax(bs, **kwargs)
        else:
            z_prior = self.prior.sample((bs,1)).to(encoded_samples.device)

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