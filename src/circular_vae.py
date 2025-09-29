# file: circular_vae.py

import torch
import torch.nn as nn
from base_vae import BaseVAE
from typing import Tuple

class CircularVAE(BaseVAE):
    """
    VAE with a von Mises latent space for circular topology.
    - q(z|x) is a von Mises distribution.
    - p(z) is a von Mises distribution (e.g., uniform on the circle).
    """
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        prior_kappa: float = 1e-2, # Low kappa = near uniform prior
        n_hidden: int = 128,
        n_layers: int = 2,
    ):
        super().__init__(n_input, n_latent, n_hidden, n_layers)
        
        # Prior parameters (uniform on circle)
        self.register_buffer("prior_mu", torch.zeros(n_latent))
        self.register_buffer("prior_kappa", torch.full((n_latent,), prior_kappa))

        # Encoder layers for von Mises parameters mu (location) and kappa (concentration)
        hidden_size = n_hidden
        # Output 2D vector and use atan2 to get an angle mu in [-pi, pi]
        self.qz_mu_vec = nn.Linear(hidden_size, n_latent * 2)
        # Output log_kappa and exponentiate to ensure kappa > 0
        self.qz_logkappa = nn.Linear(hidden_size, n_latent)

    def _get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        
        # Mean angle mu
        mu_vec = self.qz_mu_vec(hidden).view(-1, self.n_latent, 2)
        mu = torch.atan2(mu_vec[:, :, 1], mu_vec[:, :, 0])
        
        # Concentration kappa
        log_kappa = self.qz_logkappa(hidden)
        kappa = torch.exp(log_kappa)
        
        return mu, kappa

    def _reparameterize(self, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for von Mises distribution using rejection sampling.
        From: "Auto-Encoding Variational Bayes" Appendix C.2 by Kingma & Welling (2014)
        """
        # This implementation is for a single dimension at a time
        # We can batch this with careful tensor manipulation
        batch_size, dim = mu.shape
        mu_flat = mu.flatten()
        kappa_flat = kappa.flatten()
        
        samples = torch.zeros_like(mu_flat)
        
        for i in range(len(mu_flat)):
            samples[i] = self._sample_von_mises_one_dim(mu_flat[i], kappa_flat[i])
            
        return samples.view(batch_size, dim)

    def _sample_von_mises_one_dim(self, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """Helper for single scalar sampling."""
        # Algorithm from "Statistical Distributions" by Forbes et al.
        tau = 1.0 + torch.sqrt(1.0 + 4.0 * kappa**2)
        rho = (tau - torch.sqrt(2.0 * tau)) / (2.0 * kappa)
        r = (1.0 + rho**2) / (2.0 * rho)

        while True:
            u1 = torch.rand_like(mu)
            u2 = torch.rand_like(mu)
            z = torch.cos(torch.pi * u1)
            f = (1.0 + r * z) / (r + z)
            c = kappa * (r - f)
            
            if (u2 < c * (2.0 - c)) or (torch.log(c) - torch.log(u2) + 1.0 - c >= 0):
                break
        
        u3 = torch.rand_like(mu)
        sample = mu + torch.sign(u3 - 0.5) * torch.acos(f)
        sample = torch.fmod(sample + torch.pi, 2 * torch.pi) - torch.pi # wrap to [-pi, pi]
        return sample
    
    def _kl_divergence(self, mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """
        Analytical KL divergence between two von Mises distributions.
        KL(q(z|x) || p(z))
        Requires torch.special.i0 (modified Bessel function of order 0)
        """
        # Ensure i0 is available
        try:
            from torch.special import i0
        except ImportError:
            raise ImportError("Please update PyTorch to a version with `torch.special.i0`")

        kl = (
            kappa * (i0(self.prior_kappa) / i0(kappa)) * torch.cos(mu - self.prior_mu)
            - kappa
            + self.prior_kappa
            - (torch.log(i0(self.prior_kappa)) - torch.log(i0(kappa)))
        )
        return kl.sum(dim=-1)