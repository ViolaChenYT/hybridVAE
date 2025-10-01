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
        likelihood: str = "nb",
        gaussian_homoscedastic: bool = False,   
        n_batches: int = 0,
        batch_emb_dim: int = 8,
    ):
        super().__init__()
        assert likelihood in ["nb", "gaussian"], "likelihood must be 'nb' or 'gaussian'"
        self.n_input = n_input
        self.n_latent = n_latent
        self.likelihood = likelihood
        self.gaussian_homoscedastic = gaussian_homoscedastic

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

        self.n_batches = int(n_batches)
        self.batch_emb_dim = int(batch_emb_dim)
        if self.n_batches > 0:
            self.batch_emb = nn.Embedding(self.n_batches, self.batch_emb_dim)
            # adapt encoder features: [h, b] -> h̃  (same width as h)
            self.enc_cond = nn.Linear(n_hidden + self.batch_emb_dim, n_hidden)
            # adapt decoder input:   [z, b] -> z̃  (same width as z)
            self.dec_cond = nn.Linear(self.n_latent + self.batch_emb_dim, self.n_latent)

        # Output layers for likelihoods
        if self.likelihood == "nb":
            # Negative Binomial: mean and inverse dispersion
            self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_input), nn.Softmax(dim=-1))
            self.px_r_decoder = nn.Linear(n_hidden, n_input)
            # Remove any Gaussian-specific attributes if present
            self.px_mu_decoder = None
            self.px_logvar_decoder = None
        elif self.likelihood == "gaussian":
            self.px_mu_decoder = nn.Linear(n_hidden, n_input)
            if not self.gaussian_homoscedastic:
                self.px_logvar_decoder = nn.Linear(n_hidden, n_input)
            else:
                self.px_logvar_decoder = None
            # Remove any NB-specific attributes if present
            self.px_scale_decoder = None
            self.px_r_decoder = None
        else:
            raise ValueError(f"Unknown likelihood: {self.likelihood}")


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
    def _get_latent_params_from_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, ...]:
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

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor | None = None) -> dict:
        """
        Forward pass through the VAE.
        """
        # 1. Get library size factor for NB loss
        library = None
        if self.likelihood == "nb":
            library = torch.sum(x, dim=1, keepdim=True).clamp_min(1e-8)
        # 2) shared encoder features (optionally batch-conditioned)
        h = self.encoder(x)  # (B, n_hidden)
        b = None
        if (self.n_batches > 0) and (batch_index is not None):
            if batch_index.dtype != torch.long:
                batch_index = batch_index.long()
            b = self.batch_emb(batch_index)  # (B, E)
            h = self.enc_cond(torch.cat([h, b], dim=1))  # (B, n_hidden)

        # 3) latent params
        try:
            latent_params = self._get_latent_params_from_h(h)
        except NotImplementedError:
            # Fallback: subclass didn’t implement the hook; it will call self.encoder(x) internally.
            latent_params = self._get_latent_params(x)
        
        # 3. Sample from latent space
        z = self._reparameterize(*latent_params)
        
        z_in = z if b is None else self.dec_cond(torch.cat([z, b], dim=1))
        hidden_decoder = self.decoder(z_in)

        out = {"z": z, "latent_params": latent_params}
        
        # 5. Get reconstruction distribution parameters
        if self.likelihood == "nb":
            # Mean of the NB distribution
            px_scale = self.px_scale_decoder(hidden_decoder)
            px_rate = library * px_scale
            # Inverse dispersion of the NB distribution
            #px_r = torch.exp(self.px_r_decoder(hidden_decoder))
            px_r = torch.nn.functional.softplus(self.px_r_decoder(hidden_decoder)) + 1e-6
            px_r = px_r.clamp(min=1e-6, max=1e6)  
            px_rate = px_rate.clamp(min=1e-8, max=1e12)
            out.update({"px_rate": px_rate, "px_r": px_r})
        else:
            px_mu = self.px_mu_decoder(hidden_decoder)
            if self.gaussian_homoscedastic:
                out.update({"px_mu": px_mu, "px_logvar": None})
            else:
                px_logvar = self.px_logvar_decoder(hidden_decoder)
                px_logvar = torch.clamp(px_logvar, min=-10.0, max=10.0) 
                out.update({"px_mu": px_mu, "px_logvar": px_logvar})
        
        return out

    def loss(self, x: torch.Tensor, forward_output: dict, kl_weight: float = 1.0) -> dict:
        """
        Calculates the evidence lower bound (ELBO) loss.
        """
        # --- CORRECTED PART ---
        eps = 1e-8

        if self.likelihood == "nb":
            px_rate = forward_output["px_rate"]
            px_r = forward_output["px_r"]
            
            # Add a small constant for numerical stability to prevent log(0)
            #eps = 1e-8
            
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
        else:
            mu = forward_output["px_mu"]
            if self.gaussian_homoscedastic or (forward_output.get("px_logvar", None) is None):
                recon_loss = ((x - mu) ** 2).sum(dim=-1) * 0.5
            else:
                logvar = forward_output["px_logvar"]
                var = torch.exp(logvar).clamp_min(1e-10)
                #recon_loss = 0.5 * (logvar + (x - mu) ** 2 / var + torch.log(torch.tensor(2 * 3.141592653589793, device=x.device)))
                recon_loss = 0.5 * (logvar + (x - mu) ** 2 / var).sum(dim=-1)


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