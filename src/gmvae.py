# file: gmvae.py
import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from base_vae import BaseVAE
from typing import Tuple, Dict

class GMVAE(BaseVAE):
    """
    Real GMVAE (fixed-mean prior in latent space): Rui Shu's explanation
      c ~ Cat(pi)
      z | c ~ N(mu_c, sigma_p^2 I)         # mu_c fixed prototypes
      x | z ~ p_phi(x | z)                 # your decoder + likelihood
    Inference:
      q(c|x) = Cat(logits_c(x))
      q(z|x,c) = N(mu_q(x,c), diag(var_q(x,c)))
    ELBO terms:
      KL_c = KL(q(c|x) || p(c))
      KL_z = E_{q(c|x)} KL(q(z|x,c) || N(mu_c, sigma_p^2 I))
    """
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        fixed_means: torch.Tensor, # Shape: (n_components, n_latent)
        prior_sigma: float = 0.5, # shared across all clusters
        n_batches: int = 0,
        batch_emb_dim: int = 8,
        batch_index: torch.Tensor | None = None,
        n_hidden: int = 128,
        n_layers: int = 2,
        prior_logits: torch.Tensor | None = None,  # optional non-uniform p(c)
    ):
        super().__init__(n_input, n_latent, n_hidden, n_layers, n_batches, batch_emb_dim)

        n_components, d = fixed_means.shape
        assert d == n_latent, "fixed_means dim must equal n_latent"
        self.n_components = n_components
        self.n_latent = n_latent
        
        # Register fixed means and other prior parameters
        self.register_buffer("mu_prior", fixed_means)
        self.register_buffer("log_sigma_p", torch.tensor(math.log(prior_sigma)))
        p_logits = torch.zeros(self.n_components) if prior_logits is None else prior_logits
        self.register_buffer("prior_cat_logits", p_logits)

        # Encoder layers for mu and log_var
        # q(c|x)
        self.enc_c = nn.Linear(n_hidden, n_components)
        # q(z|x,c)
        self.enc_z = nn.Linear(n_hidden, 2 * n_latent * n_components)
        # placeholder for extra nonlinearity???
        self.qz_act = nn.Identity()  # placeholder for extra nonlinearity???


    def _get_latent_params_from_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  
        logits_c = self.enc_c(h)  # (B, K)
        raw = self.enc_z(h).view(h.size(0), self.n_components, 2 * self.n_latent)
        mu_q, logvar_q = torch.chunk(raw, 2, dim=-1)
        logvar_q = logvar_q.clamp(-10.0, 10.0)
        return logits_c, mu_q, logvar_q

    def _get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self._get_latent_params_from_h(hidden)

    def _reparameterize(
            self,
            logits_c: torch.Tensor,
            mu_q: torch.Tensor,
            logvar_q: torch.Tensor,
        ) -> torch.Tensor:
            """
            Sample c ~ q(c|x), then z ~ q(z|x,c). Return z to the decoder.

            Inputs:
            logits_c : (B, K)
            mu_q     : (B, K, d)
            logvar_q : (B, K, d)
            Output:
            z        : (B, d)
            """
            q_c = Categorical(logits=logits_c)                     # distribution over components
            c = q_c.sample()                                       # (B,)
            batch_size = logits_c.shape[0]
            idx = c.view(batch_size, 1, 1).expand(batch_size, 1, self.n_latent)      # (B, 1, d)

            mu_q_s = torch.gather(mu_q, 1, idx).squeeze(1)         # (B, d)
            logvar_q_s = torch.gather(logvar_q, 1, idx).squeeze(1) # (B, d)
            std_q_s = torch.exp(0.5 * logvar_q_s)

            eps = torch.randn_like(std_q_s)
            z = mu_q_s + std_q_s * eps                             # (B, d)
            return z

    def _kl_divergence(
        self,
        logits_c: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q(c|x)||p(c)) + E_{q(c|x)} KL(q(z|x,c) || N(mu_c, sigma_p^2 I)).
        Returns shape (B,).
        """
        # ---- KL for the categorical ----
        q_c = Categorical(logits=logits_c)
        p_c = Categorical(logits=self.prior_cat_logits.expand_as(logits_c))
        kl_c = torch.distributions.kl.kl_divergence(q_c, p_c)                  # (B,)

        # ---- Expected KL between Gaussians per component (closed form) ----
        var_q = torch.exp(logvar_q)                                            # (B, K, d)
        logvar_p = 2.0 * self.log_sigma_p                                      # scalar tensor
        var_p = torch.exp(logvar_p)                                            # scalar
        mu_p = self.mu_prior.unsqueeze(0)                                      # (1, K, d)

        # 0.5 * sum_d [ log(var_p/var_q) + (var_q + (mu_q - mu_p)^2)/var_p - 1 ]
        term = (logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0
        kl_z_given_c = 0.5 * term.sum(dim=-1)                                  # (B, K)

        probs_c = q_c.probs                                                    # (B, K)
        kl_z = (probs_c * kl_z_given_c).sum(dim=1)                             # (B,)

        return kl_c + kl_z