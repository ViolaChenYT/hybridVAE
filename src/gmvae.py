# file: gmvae.py
import math
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from base_vae import BaseVAE
from typing import Tuple, Dict
import pdb

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
        n_components: int,
        fixed_means: torch.Tensor | None = None, # Shape: (n_components, n_latent)
        prior_sigma: float | None = None, # shared across all clusters
        n_batches: int = 0,
        batch_emb_dim: int = 8,
        batch_index: torch.Tensor | None = None,
        likelihood: str = "nb",
        n_hidden: int = 128,
        n_layers: int = 2,
        prior_logits: torch.Tensor | None = None,  # optional non-uniform p(c)
    ):
        super().__init__(n_input, n_latent, n_hidden, n_layers, likelihood, n_batches=n_batches, batch_emb_dim=batch_emb_dim)

        self.n_components = n_components
        self.n_latent = n_latent
        self.prior_sigma = prior_sigma
        self.fixed_means = fixed_means
        if self.fixed_means is not None:
            n_parts, d = fixed_means.shape
            assert d == n_latent, "fixed_means dim must equal n_latent"
            assert n_parts == n_components, "fixed_means rows must equal n_components"
        
        # Register fixed means and other prior parameters
        #self.register_buffer("mu_prior", fixed_means)

        self.c_encoder = self._build_network(
            n_in=1,
            n_out=32,
            n_hidden=32,
            n_layers=1,
            activation=nn.ReLU()
        )
        self.p_zgivenc = nn.Linear(32, 2*n_latent)
        if fixed_means is not None:
            self.register_buffer("mu_prior", fixed_means)
        if prior_sigma is not None:
            self.register_buffer("log_sigma_p", torch.tensor(math.log(prior_sigma)))
        '''
        if prior_sigma is None and fixed_means is None:
            #self.register_buffer("log_sigma_p", torch.tensor(math.log(prior_sigma)))
            self.c_encoder = self._build_network(
                n_in=1,
                n_out=32,
                n_hidden=32,
                n_layers=1,
                activation=nn.ReLU()
            )
            self.p_zgivenc = nn.Linear(32, 2*n_latent)
        else:
            self.register_buffer("mu_prior", fixed_means)
            self.register_buffer("log_sigma_p", torch.tensor(math.log(prior_sigma)))
        '''
        p_logits = torch.zeros(self.n_components) if prior_logits is None else prior_logits
        self.register_buffer("prior_cat_logits", p_logits)

        # Encoder layers for mu and log_var
        # q(c|x)
        self.enc_c = nn.Linear(n_hidden, n_components)
        # q(z|x,c)
        self.enc_z = nn.Linear(n_hidden, 2 * n_latent * n_components)
        # placeholder for extra nonlinearity???
        self.qz_act = nn.Identity()  # placeholder for extra nonlinearity???
        
        # Initialize enc_c for uniform component usage
        with torch.no_grad():
            self.enc_c.weight.zero_()
            self.enc_c.bias.zero_()  # for uniform p(c); if non-uniform, set to log π


    def _get_latent_params_from_h(self, h: torch.Tensor, residual_mean_pen = False, hard_fix = False) -> Tuple[torch.Tensor, torch.Tensor]:  
        logits_c = self.enc_c(h)  # (B, K)
        raw = self.enc_z(h).view(h.size(0), self.n_components, 2 * self.n_latent)
        mu_q, logvar_q = torch.chunk(raw, 2, dim=-1)
        logvar_q = logvar_q.clamp(-10.0, 10.0)
        if hard_fix:
            mu_q = self.mu_prior.unsqueeze(0).expand(h.size(0), self.n_components, self.n_latent)  # fixed mean
        elif residual_mean_pen:
            mu_q = mu_q + self.mu_prior.unsqueeze(0)
        return logits_c, mu_q, logvar_q

    def _get_latent_params(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self._get_latent_params_from_h(hidden)

    def _reparameterize(self, logits_c, mu_q, logvar_q):
        # Moment-matched Gaussian for q(z|x) = sum_k q_k N(mu_q, diag(var_q))
        probs = torch.softmax(logits_c, dim=-1)                     # (B,K)
        m = (probs.unsqueeze(-1) * mu_q).sum(dim=1)                 # (B,d)
        second = (probs.unsqueeze(-1) * (torch.exp(logvar_q) + mu_q**2)).sum(dim=1)
        v = (second - m**2).clamp_min(1e-8)
        eps = torch.randn_like(m)
        return m + torch.sqrt(v) * eps

    def _kl_divergence_split(self, logits_c, mu_q, logvar_q):
        """Return (kl_c, kl_z) separately, each shape (B,)."""
        q_c = torch.distributions.Categorical(logits=logits_c)
        p_c = torch.distributions.Categorical(
            logits=self.prior_cat_logits.expand_as(logits_c)
        )
        kl_c = torch.distributions.kl.kl_divergence(q_c, p_c)            # (B,)

        # Gaussian expected KL same as your _kl_divergence
        var_q  = torch.exp(logvar_q)                                     # (B,K,d)
        #pdb.set_trace()
        c_input = torch.tensor([i for i in range(self.n_components)], device=logits_c.device).unsqueeze(1).float()  # (K,1)
        h_c = self.c_encoder(c_input)  # (K,32)
        mu_p, logvp = torch.chunk(self.p_zgivenc(h_c), 2, dim=-1)  # each (K,d)
        mu_p = mu_p.unsqueeze(0)                                         # (1,K,d)
        logvp = logvp.clamp(-10.0, 10.0).unsqueeze(0)           # (1,K,d)
        var_p = torch.exp(logvp)                                 # (1,K,d)
        if self.prior_sigma is not None:
            logvp  = 2.0 * self.log_sigma_p                                  # scalar
            var_p  = torch.exp(logvp)
        if self.fixed_means is not None:
            mu_p   = self.mu_prior.unsqueeze(0)                               # (1,K,d)
        '''
        if self.prior_sigma is None and self.fixed_means is None:
            c_input = torch.tensor([i for i in range(self.n_components)], device=logits_c.device).unsqueeze(1).float()  # (K,1)
            h_c = self.c_encoder(c_input)  # (K,32)
            mu_p, logvp = torch.chunk(self.p_zgivenc(h_c), 2, dim=-1)  # each (K,d)
            mu_p = mu_p.unsqueeze(0)                                         # (1,K,d)
            logvp = logvp.clamp(-10.0, 10.0).unsqueeze(0)           # (1,K,d)
            var_p = torch.exp(logvp)                                 # (1,K,d)
        else:
            logvp  = 2.0 * self.log_sigma_p                                  # scalar
            var_p  = torch.exp(logvp)
            mu_p   = self.mu_prior.unsqueeze(0)                               # (1,K,d)
        '''
        term   = (logvp - logvar_q) + (var_q + (mu_q - mu_p)**2)/var_p - 1.0
        kl_z_given_c = 0.5 * term.sum(dim=-1)                             # (B,K)

        probs = q_c.probs                                                 # (B,K)
        kl_z  = (probs * kl_z_given_c).sum(dim=1)                         # (B,)

        #print("KL components:")
        #print(kl_c.mean().item(), kl_z.mean().item())

        return kl_c, kl_z

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
        kl_c, kl_z = self._kl_divergence_split(logits_c, mu_q, logvar_q)
        return kl_c + kl_z
    def _expected_recon_over_c(self, x, logits_c, mu_q, logvar_q):
        """
        Computes E_{q(c|x)} E_{q(z|x,c)} [ -log p(x|z) ]
        by enumerating all K components (low variance).
        Returns per-sample recon loss, shape (B,).
        """
        B, K, d = mu_q.shape
        device = x.device

        # responsibilities
        probs = torch.softmax(logits_c, dim=-1)                 # (B,K)

        # sample z for each component (1 sample per k is plenty here)
        std = torch.exp(0.5 * logvar_q)                         # (B,K,d)
        eps = torch.randn_like(std)
        z_k = mu_q + std * eps                                   # (B,K,d)

        # decode each (B,K,d) → (B*K, d) → decode → NB params → reshape back
        z_flat = z_k.reshape(B * K, d)
        h_dec  = self.decoder(z_flat)                            # (B*K, hidden)
        px_scale = self.px_scale_decoder(h_dec)                  # (B*K, G)
        px_r     = torch.nn.functional.softplus(self.px_r_decoder(h_dec)) + 1e-6
        px_scale = px_scale.clamp(1e-8, 1.0)
        px_r     = px_r.clamp(1e-6, 1e6)

        # library sizes per sample (B,1) broadcast to (B,K,G)
        library = x.sum(dim=1, keepdim=True).clamp_min(1e-8)
        px_rate = (library.repeat(1, K)[:, :, None] * px_scale.view(B, K, -1)).clamp(1e-8, 1e12)  # (B,K,G)
        px_r    = px_r.view(B, K, -1)                                                               # (B,K,G)

        # NB log prob for each k, then expectation over c with weights probs
        # log p(x|z_k) summed over genes → (B,K)
        from torch.distributions import NegativeBinomial
        x_expand = x[:, None, :].expand_as(px_rate)                                                 # (B,K,G)
        logits = (px_r.clamp_min(1e-8)).log() - (px_rate.clamp_min(1e-8)).log()
        logp_x_given_zk = NegativeBinomial(total_count=px_r, logits=logits).log_prob(x_expand).sum(-1)  # (B,K)

        # E_{q(c|x)}[ -log p(x|z) ]  (minus sign for loss)
        recon = -(probs * logp_x_given_zk).sum(dim=1)                                               # (B,)
        return recon


    def loss(self, x, forward_output, kl_c_weight=1.0, kl_z_weight=1.0,
         aggregate_alignment_pen=False, residual_mean_pen=False,
         align_weight=5e-2, anchor_weight=1e-2):
        logits_c, mu_q, logvar_q = forward_output["latent_params"]

        # low-variance reconstruction (expected over c)
        recon_loss = self._expected_recon_over_c(x, logits_c, mu_q, logvar_q)  # (B,)

        # KL as you already derive (categorical + expected Gaussian)
        #kl_local = self._kl_divergence(logits_c, mu_q, logvar_q)               # (B,)
        kl_c, kl_z = self._kl_divergence_split(logits_c, mu_q, logvar_q)

        loss = (recon_loss + kl_c_weight * kl_c + kl_z_weight * kl_z).mean()
        #print("Loss components:")
        #print(recon_loss.mean().item(), kl_local.mean().item())
        out = {"loss": loss, "recon_loss": recon_loss.mean(), "kl_c": kl_c.mean(), "kl_z": kl_z.mean()}

        # optional regularizers you already implemented
        if residual_mean_pen:
            resid = mu_q - self.mu_prior.unsqueeze(0)
            anchor = (resid ** 2).mean()
            loss = loss + anchor_weight * anchor
            out.update({"loss": loss, "anchor": anchor.detach()})
        if aggregate_alignment_pen:
            probs = torch.softmax(logits_c, -1)
            denom = probs.sum(0, keepdim=True).clamp_min(1e-8)
            mu_bar = ((probs / denom).unsqueeze(-1) * mu_q).sum(0)
            align = ((mu_bar - self.mu_prior) ** 2).sum(-1).mean()
            loss = loss + align_weight * align
            out.update({"loss": loss, "align": align.detach()})
        return out

