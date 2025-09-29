import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from base_vae import BaseVAE
from typing import Tuple
import math
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.utils.data import DataLoader, TensorDataset

class VAE(BaseVAE):
    """
    q(z|x) = N(mu, diag(sigma^2)), p(z) = N(0, I)
    Decoder is NB via BaseVAE.forward (px_rate, px_r).
    """
    def __init__(
        self,
        n_input: int,
        n_latent: int,
        n_hidden: int = 128,
        n_layers: int = 2,
        clamp_logvar: tuple[float, float] = (-10.0, 10.0),
    ):
        super().__init__(n_input, n_latent, n_hidden=n_hidden, n_layers=n_layers)
        # Heads from the encoder's final hidden to μ and logσ²
        self.z_mean = nn.Linear(n_hidden, n_latent)
        self.z_logvar = nn.Linear(n_hidden, n_latent)
        self._clamp_lv = clamp_logvar

        # Optional: small weight init for stability
        for m in [self.z_mean, self.z_logvar, self.px_r_decoder]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ---- required by BaseVAE ----
    def _get_latent_params(self, x: torch.Tensor):
        h_enc = self.encoder(x)                     # [B, n_hidden]
        mu = self.z_mean(h_enc)                     # [B, n_latent]
        logvar = self.z_logvar(h_enc)               # [B, n_latent]
        # keep variance sane to avoid exp(logvar) overflow -> NaN
        logvar = torch.clamp(logvar, *self._clamp_lv)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL( N(mu,σ^2) || N(0,1) ) per-sample
        logvar = torch.clamp(logvar, *self._clamp_lv)
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return torch.sum(kl, dim=-1)  # [B]

    # (Optional) guardrails for decoder extremes
    def forward(self, x: torch.Tensor) -> dict:
        out = super().forward(x)
        # Cap to avoid pathological values in early training
        out["px_rate"] = torch.clamp(out["px_rate"], min=1e-8, max=1e8)
        out["px_r"]    = torch.clamp(out["px_r"],    min=1e-6, max=1e8)
        return out


@torch.no_grad()
def generate_synthetic_nb(
    n_samples=1000, n_features=50, n_latent=8, theta_val=10.0, seed=7, device="cpu"
):
    """
    Simple NB generator consistent with decoder:
    z ~ N(0,I), mu = softmax(W z + b) * library; x ~ NB(mean=mu, r=theta)
    We randomize per-cell library sizes to resemble scRNA-seq.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    W = torch.randn(n_features, n_latent, generator=g, device=device) / math.sqrt(n_latent)
    b = torch.randn(n_features, generator=g, device=device) * 0.2

    z = torch.randn(n_samples, n_latent, generator=g, device=device)
    logits = z @ W.T + b
    px_scale = F.softmax(logits, dim=-1)                        # [N,D]
    lib = torch.exp(torch.randn(n_samples, 1, generator=g, device=device) * 0.3 + 8.5) # ~ e^[~8.5] ~ ~5k
    mu = lib * px_scale                                         # [N,D]

    r = torch.full_like(mu, float(theta_val))
    p = r / (r + mu + 1e-8)                                     # NB paramization
    nb = NegativeBinomial(total_count=r, probs=p)
    x = nb.sample()
    return x  # int counts


def fit(
    model: VAE,
    data_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    kl_warmup: int = 10,
    device: str = "cpu",
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        kl_w = min(1.0, ep / max(1, kl_warmup))
        total, trecon, tkl, n = 0.0, 0.0, 0.0, 0

        for (xb,) in data_loader:
            xb = xb.to(device).float()
            fwd = model(xb)
            losses = model.loss(xb, fwd, kl_weight=kl_w)
            loss = losses["loss"]

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = xb.size(0)
            total += loss.item() * bs
            trecon += losses["recon_loss"].item() * bs
            tkl += losses["kl_local"].item() * bs
            n += bs

        log = {
            "epoch": ep,
            "loss": total / n,
            "recon": trecon / n,
            "kl": tkl / n,
            "kl_w": kl_w,
        }
        history.append(log)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{ep:03d}] loss={log['loss']:.3f}  recon={log['recon']:.3f}  kl={log['kl']:.3f} (kl_w={kl_w:.2f})")

    return history

@torch.no_grad()
def evaluate(model: VAE, x: torch.Tensor, device="cpu") -> dict:
    model.eval().to(device)
    x = x.to(device).float()
    fwd = model(x)
    losses = model.loss(x, fwd)
    return {k: (v.item() if torch.is_tensor(v) else v) for k, v in losses.items()}

@torch.no_grad()
def reconstruct(model: VAE, x: torch.Tensor, device="cpu"):
    model.eval().to(device)
    x = x.to(device).float()
    fwd = model(x)
    return fwd["px_rate"]  # NB mean μ = px_rate