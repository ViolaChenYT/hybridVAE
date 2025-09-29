# file: gmvae.py

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from base_vae import BaseVAE
from typing import Tuple

class GMVAE(BaseVAE):
    """
    Gaussian Mixture VAE with a fixed-mean prior.
    - q(z|x) is a single Gaussian.
    - p(z) is a GMM with fixed means.
    """
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        fixed_means: torch.Tensor, # Shape: (n_components, n_latent)
        prior_var: float = 1.0,
        n_hidden: int = 128,
        n_layers: int = 2,
    ):
        super().__init__(n_input, n_latent, n_hidden, n_layers)
        
        # Register fixed means and other prior parameters
        self.register_buffer("fixed_means", fixed_means)
        self.n_components = fixed_means.shape[0]
        self.register_buffer("prior_var", torch.full((self.n_components,), prior_var))
        
        # Uniform categorical distribution over components
        self.register_buffer("prior_cat_probs", torch.ones(self.n_components) / self.n_components)

        # Encoder layers for mu and log_var
        hidden_size = n_hidden
        self.qz_mean = nn.Linear(hidden_size, n_latent)
        self.qz_logvar = nn.Linear(hidden_size, n_latent)

    def _get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        mu = self.qz_mean(hidden)
        log_var = self.qz_logvar(hidden)
        return mu, log_var

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Calculates KL(q(z|x) || p(z)) where p(z) is a GMM.
        We use Monte Carlo approximation: E_q[log q(z|x) - log p(z)]
        """
        # z is a sample from q(z|x)
        z = self._reparameterize(mu, log_var)
        
        # 1. log q(z|x)
        q_dist = Normal(mu, torch.exp(0.5 * log_var))
        log_qz_x = q_dist.log_prob(z).sum(dim=1)
        
        # 2. log p(z) = log(sum_k [p(c=k) * p(z|c=k)])
        # p(z|c=k) are the fixed Gaussian components
        p_dist = Normal(self.fixed_means, self.prior_var.sqrt().unsqueeze(-1))
        
        # log_prob for each component: (batch_size, n_components)
        log_pz_c = p_dist.log_prob(z.unsqueeze(1)).sum(dim=2)
        
        # log p(c=k)
        log_pc = self.prior_cat_probs.log()
        
        # Total log p(z) using log-sum-exp trick for stability
        log_pz = torch.logsumexp(log_pc + log_pz_c, dim=1)
        
        # KL divergence
        return log_qz_x - log_pz