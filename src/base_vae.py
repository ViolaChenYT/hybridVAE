# file: base_vae.py

import torch
import torch.nn as nn
from torch.distributions import NegativeBinomial
from abc import ABC, abstractmethod
from typing import List, Tuple

class BaseVAE(nn.Module, ABC):
    """
    A generic base class for Variational Autoencoders.
    Subclasses must implement the latent space specific methods.
    """
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_hidden: int = 64,
        n_layers: int = 2,
    ):
        super().__init__()
        self.n_input = n_input
        self.n_latent = n_latent

        # Encoder and Decoder networks
        self.encoder = self._build_network(
            n_in=n_input, 
            n_out=n_hidden, # Encoder's final layer will be handled in subclass
            n_hidden=n_hidden, 
            n_layers=n_layers
        )
        # The latent-specific output layers are defined in subclasses
        
        self.decoder = self._build_network(
            n_in=n_latent,
            n_out=n_hidden,
            n_hidden=n_hidden,
            n_layers=n_layers,
            activation=nn.ReLU()
        )
        
        # Output layers for Negative Binomial distribution
        # Mean parameter
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_input), nn.Softmax(dim=-1))
        # Inverse dispersion parameter
        self.px_r_decoder = nn.Linear(n_hidden, n_input)


    def _build_network(
        self, 
        n_in: int, 
        n_out: int, 
        n_hidden: int, 
        n_layers: int, 
        activation: nn.Module = nn.ReLU()
    ) -> nn.Module:
        """Helper to build a simple MLP."""
        layers = [nn.Linear(n_in, n_hidden), activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(n_hidden, n_hidden), activation])
        layers.append(nn.Linear(n_hidden, n_out))
        return nn.Sequential(*layers)

    @abstractmethod
    def _get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Encodes input to the parameters of the latent distribution.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _reparameterize(self, *params) -> torch.Tensor:
        """
        Samples from the latent space using the reparameterization trick.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def _kl_divergence(self, *params) -> torch.Tensor:
        """
        Calculates the KL divergence between the posterior and the prior.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the VAE.
        """
        # 1. Get library size factor for NB loss
        library = torch.sum(x, dim=1, keepdim=True)
        
        # 2. Encode and get latent parameters
        latent_params = self._get_latent_params(x)
        
        # 3. Sample from latent space
        z = self._reparameterize(*latent_params)
        
        # 4. Decode
        hidden_decoder = self.decoder(z)
        
        # 5. Get reconstruction distribution parameters
        # Mean of the NB distribution
        px_scale = self.px_scale_decoder(hidden_decoder)
        px_rate = library * px_scale
        # Inverse dispersion of the NB distribution
        px_r = torch.exp(self.px_r_decoder(hidden_decoder))

        return {
            "z": z,
            "latent_params": latent_params,
            "px_rate": px_rate,
            "px_r": px_r,
        }

    def loss(self, x: torch.Tensor, forward_output: dict, kl_weight: float = 1.0) -> dict:
        """
        Calculates the evidence lower bound (ELBO) loss.
        """
        # --- CORRECTED PART ---

        px_rate = forward_output["px_rate"]
        px_r = forward_output["px_r"]
        
        # Add a small constant for numerical stability to prevent log(0)
        eps = 1e-8
        
        logits = torch.log(px_r + eps) - torch.log(px_rate + eps)

        # --- DIAGNOSTIC PRINTS to add ---
        if torch.isnan(px_rate).any() or torch.isinf(px_rate).any():
            print(f"!!! px_rate contains nan or inf. Max value: {px_rate.max()}")
        if torch.isnan(px_r).any() or torch.isinf(px_r).any():
            print(f"!!! px_r contains nan or inf. Max value: {px_r.max()}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"!!! logits contains nan or inf. Max value: {logits.max()}")
        # --- END DIAGNOSTICS ---
        
        # Reconstruction Loss (Negative Log-Likelihood of NB)
        recon_loss = -NegativeBinomial(
            total_count=px_r, 
            logits=logits
        ).log_prob(x).sum(dim=-1)

        # --- END OF CORRECTION ---

        # KL Divergence
        kl_local = self._kl_divergence(*forward_output["latent_params"])
        
        # Total Loss
        loss = torch.mean(recon_loss + kl_weight * kl_local)
        
        return {
            "loss": loss,
            "recon_loss": recon_loss.mean(),
            "kl_local": kl_local.mean(),
        }