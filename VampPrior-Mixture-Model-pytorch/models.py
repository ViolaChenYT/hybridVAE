import os
import torch
import torch.nn as nn
import numpy as np

import math
import abc
from typing import Dict, Any, Tuple, Optional
import torch.nn.functional as F
import torch.distributions as td
from contextlib import contextmanager
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def build_encoder(
    dim_x: int,
    h_dim: int,
    n_layers: int,
    dropout: float = 0.0,
    *,
    use_batchnorm: bool = True,
    bn_momentum: float = 0.1,
    bn_eps: float = 1e-5,
    device=None,
):
    layers = []
    in_dim = dim_x
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, h_dim, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h_dim, eps=bn_eps, momentum=bn_momentum))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = h_dim
    return nn.Sequential(*layers) if device is None else nn.Sequential(*layers).to(device)


class DiagGaussianDecoder(nn.Module):
    def __init__(self, dim_x: int, latent_dim: int, h_dim: int, n_layers: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(latent_dim, h_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        # Two heads: mean and log-variance (diagonal)
        self.mu_head = nn.Linear(h_dim, dim_x)
        self.logvar_head = nn.Linear(h_dim, dim_x)

        # (Optional) small init for stability on logvar
        nn.init.constant_(self.logvar_head.bias, -4.0)  # start with std ~ exp(-2) ≈ 0.135

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        h = self.backbone(z)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        std = (0.5 * logvar).exp().clamp_min(1e-8)
        base = td.Normal(loc=mu, scale=std)
        return td.Independent(base, reinterpreted_batch_ndims=1)


def build_decoder_gaussian(dim_x, latent_dim, h_dim, n_layers, dropout=0.0, device=None):
    dec = DiagGaussianDecoder(dim_x=dim_x, latent_dim=latent_dim, h_dim=h_dim, n_layers=n_layers, dropout=dropout)
    if device is not None:
        dec = dec.to(device)
    return dec


class NBDecoder(nn.Module):
    """
    p(x|z) = NB(mean = mu, inv_dispersion = theta)

    PyTorch's NegativeBinomial is parameterized by (total_count=r, probs/logits).
    Mapping from (mu, theta) to (r, logits):
        r = theta
        p = r / (r + mu)
        logits = log(p / (1 - p)) = log(r) - log(mu)

    We output:
      mu    = softplus(W_mu h + b)         (per-sample, per-feature)
      theta = softplus(W_th h + b)         (per-sample, per-feature)
    """
    def __init__(
        self,
        dim_x: int,
        latent_dim: int,
        h_dim: int,
        n_layers: int,
        dropout: float = 0.0,
        eps: float = 1e-8,             # numerical floor for positivity/logs
        clamp_logit: float | None = 20, # optional clamp for logits magnitude
        theta_mode: str = "per_sample",     # "per_sample" | "per_feature" | "scalar"
        theta_init: float = 0.5,            # initial positive value for learnable theta
        use_batchnorm: bool = True,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        library_init: float = 4000.0,
    ):
        super().__init__()
        assert theta_mode in {"per_sample", "per_feature", "scalar"}

        self.softplus = nn.Softplus()
        self.eps = eps
        self.clamp_logit = clamp_logit
        self.dim_x = dim_x
        self.theta_mode = theta_mode
        #self.library = 5000.0
        self._library_u = nn.Parameter(softplus_inv(torch.tensor(library_init)))

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for layer_idx in range(n_layers):
            # If BN is used, we can drop bias in Linear (BN beta covers shift)
            layers.append(nn.Linear(in_dim, h_dim, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim, eps=bn_eps, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Two heads: mu (mean) and theta (inverse dispersion / total_count)
        self.mu_head = nn.Linear(h_dim, dim_x)
        #self.theta_head = nn.Linear(h_dim, dim_x)

        # Small negative bias so softplus starts near small positive values
        nn.init.constant_(self.mu_head.bias, -2.0)
        #nn.init.constant_(self.theta_head.bias, -2.0)

        if self.theta_mode == "per_sample":
            self.theta_head = nn.Linear(h_dim, dim_x)
            nn.init.constant_(self.theta_head.bias, -2.0)
        elif self.theta_mode == "per_feature":
            init_u = softplus_inv(torch.full((dim_x,), float(theta_init)))
            self.theta_param = nn.Parameter(init_u)  # shape [D]
        else:
            init_u = softplus_inv(torch.tensor(float(theta_init)))
            self.theta_scalar = nn.Parameter(init_u)           # shape []

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> td.Independent:
        """
        Returns an Independent(NegativeBinomial) over the last dimension (features).
        - Input:  z  [B, latent_dim]
        - Output: dist over x in shape [B, dim_x]
        """
        B = z.shape[0]
        h = self.backbone(z)

        # Positivity via softplus
        #mu = self.softplus(self.mu_head(h)) + self.eps          # [B, D]
        #library = self.softplus(self._library_u)  # scalar > 0
        library = x.sum(dim=-1, keepdim=True)  # [B, 1]
        #print(library)
        mu = F.softmax(self.mu_head(h), dim=-1) * library + self.eps  # [B, D]
        #theta = self.softplus(self.theta_head(h)) + self.eps     # [B, D]

        if self.theta_mode == "per_sample":
            theta = self.softplus(self.theta_head(h)) + self.eps        # [B, D]
        elif self.theta_mode == "per_feature":
            theta = self.softplus(self.theta_param).unsqueeze(0)         # [1, D]
            theta = theta.expand(B, -1) + self.eps                       # [B, D]
        else:
            theta = self.softplus(self.theta_scalar).view(1, 1)          # [1,1]
            theta = theta.expand(B, self.dim_x) + self.eps               # [B, D]

        # logits = log(r) - log(mu), where r = theta
        logits = torch.log(theta) - torch.log(mu)
        if self.clamp_logit is not None:
            logits = logits.clamp(min=-self.clamp_logit, max=self.clamp_logit)

        base = td.NegativeBinomial(total_count=theta, logits=logits)  # batch shape [B, D], event shape ()
        # Reinterpret the last batch dim (features) as event dims to form product distribution over D
        dist = td.Independent(base, reinterpreted_batch_ndims=1)
        return dist

def build_decoder_nb(dim_x, latent_dim, h_dim, n_layers, dropout=0.0, device=None, **kwargs):
    dec = NBDecoder(dim_x=dim_x, latent_dim=latent_dim, h_dim=h_dim, n_layers=n_layers, dropout=dropout, **kwargs)
    if device is not None:
        dec = dec.to(device)
    return dec

# ---------- tiny metric helpers ----------
class RunningMean:
    def __init__(self, dtype=torch.float32, device=None):
        self._dtype = dtype
        self._device = device
        self.reset()

    @torch.no_grad()
    def update(self, x: torch.Tensor | float):
        # Accept tensor or Python number
        if isinstance(x, torch.Tensor):
            v = x.detach().to(device=self._sum.device, dtype=self._sum.dtype)
            self._sum += v.sum()
            self._count += torch.tensor(v.numel(), device=self._sum.device, dtype=self._sum.dtype)
        else:
            # Python scalar -> convert on the buffer's device/dtype
            self._sum += torch.tensor(float(x), device=self._sum.device, dtype=self._sum.dtype)
            self._count += torch.tensor(1.0, device=self._sum.device, dtype=self._sum.dtype)

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return self._sum / self._count.clamp_min(1.0)

    @torch.no_grad()
    def reset(self):
        dev = self._device if self._device is not None else torch.device("cpu")
        self._sum   = torch.tensor(0.0, device=dev, dtype=self._dtype)
        self._count = torch.tensor(0.0, device=dev, dtype=self._dtype)

    # Optional convenience: move buffers like a module
    def to(self, device=None, dtype=None):
        if device is not None:
            self._sum = self._sum.to(device)
            self._count = self._count.to(device)
        if dtype is not None:
            self._sum = self._sum.to(dtype=dtype)
            self._count = self._count.to(dtype=dtype)
        return self

# ---------- utils ----------
def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # numerically stable inverse of softplus
    return y + torch.log(-torch.expm1(-y))

def vec_to_tril(x: torch.Tensor, D: int) -> torch.Tensor:
    """
    Map a (..., D + D*(D+1)/2) vector to (..., D, D):
      first D entries = loc
      remaining = unconstrained lower-tri (we softplus the diag)
    Returns (loc, scale_tril) with positive diagonal.
    """
    loc, tril_flat = torch.split(x, [D, D*(D+1)//2], dim=-1)
    # fill strictly lower-tri by index
    L = x.new_zeros(*x.shape[:-1], D, D)
    # indices for lower-tri (including diag)
    idx = torch.tril_indices(row=D, col=D, offset=0, device=x.device)
    L[..., idx[0], idx[1]] = tril_flat
    # softplus on diagonal to ensure positive
    diag = torch.diagonal(L, dim1=-2, dim2=-1)
    diag_sp = F.softplus(diag) + 1e-8
    L = L.clone()
    L.diagonal(dim1=-2, dim2=-1).copy_(diag_sp)
    return loc, L

class MVNTriLHead(nn.Module):
    """Outputs a MultivariateNormal with TriL scale from a hidden vector."""
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        out_dim = latent_dim + latent_dim * (latent_dim + 1) // 2
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor) -> td.MultivariateNormal:
        loc, scale_tril = vec_to_tril(self.proj(h), self.latent_dim)
        return td.MultivariateNormal(loc=loc, scale_tril=scale_tril)

class DiagGaussianHead(nn.Module):
    """Outputs q(z|x)=Independent(Normal(mu, sigma)), diagonal in latent_dim."""
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(in_dim, 2 * latent_dim)  # -> [mu | logvar]

    def forward(self, h: torch.Tensor) -> td.Independent:
        mu, logvar = torch.tensor_split(self.proj(h), 2, dim=-1)
        std = (0.5 * logvar).exp().clamp_min(1e-8)
        base = td.Normal(loc=mu, scale=std)      # event dim = 1 below
        return td.Independent(base, 1)           # shape: (B,), event_shape=(latent_dim,)

# ---------- main module ----------
class VariationalAutoencoder(nn.Module, metaclass=abc.ABCMeta):
    """
    PyTorch version of your TF VAE.
    Assumptions:
      - `encoder`: nn.Module mapping x -> hidden (last dim = enc_out_dim)
      - `decoder`: nn.Module mapping z -> x̂ (same shape as x)
      - `prior`:   object with .latent_dim, .pz(...), .kl_divergence(qz_x, encoder=...), and optional .cluster_probabilities(...)
    """

    def __init__(self, encoder: nn.Module, enc_out_dim: int, decoder: nn.Module, prior, **kwargs):
        super().__init__()
        self.prior = prior
        self.decoder = decoder

        # q(z|x) head: replace tfpl.MultivariateNormalTriL
        #self.qz_head = MVNTriLHead(in_dim=enc_out_dim, latent_dim=prior.latent_dim)
        self.qz_head = DiagGaussianHead(in_dim=enc_out_dim, latent_dim=prior.latent_dim)
        self.encoder = encoder

        # scale ~ Softplus(raw_scale); init so softplus(raw)=1.0
        #raw_init = softplus_inv(torch.tensor(1.0))
        #self.raw_scale = nn.Parameter(raw_init)

        # trackers
        self.elbo_tracker = RunningMean()
        self.ell_tracker  = RunningMean()
        self.dkl_tracker  = RunningMean()

        # clustering trackers (scalar running means; you can plug real metrics later)
        self.cluster_util = RunningMean()
        self.accuracy_tracker = RunningMean()
        self.ari_tracker = RunningMean()
        self.nmi_tracker = RunningMean()

    # ----- properties -----
    '''
    @property
    def scale(self) -> torch.Tensor:
        return F.softplus(self.raw_scale) + 1e-8
    '''

    # ----- q(z|x) definition (encoder -> MVN) -----
    def _define_variational_family(self, x: torch.Tensor, **kwargs):
        h = self.encoder(x, **kwargs)                      # (B, H)
        qz_x = self.qz_head(h)                             # MVN(loc, scale_tril)
        return qz_x

    # ----- p(x|z) = Normal(decoder(z), scale) with all but first dim as event -----
    '''
    def px(self, z: torch.Tensor, x_shape: torch.Size) -> td.Independent:
        out = self.decoder(z)

        # Case A: decoder returns (mean, logvar)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            loc, logvar = out
        else:
            # Case B: decoder returns a single tensor with last-dim = 2 * channels/features
            # Split along the last dimension into (mean, logvar)
            D_last = out.shape[-1]
            assert D_last % 2 == 0, (
                "decoder(z) must return (mean, logvar) or a tensor whose last "
                "dimension equals 2×channels/features to split into (mean, logvar)."
            )
            loc, logvar = torch.chunk(out, 2, dim=-1)

        # Diagonal std; stable parametrization via log-variance
        std = (0.5 * logvar).exp().clamp_min(1e-8)

        base = td.Normal(loc=loc, scale=std)
        reinterpreted_batch_ndims = loc.dim() - 1          # treat all non-batch dims as event dims
        return td.Independent(base, reinterpreted_batch_ndims=reinterpreted_batch_ndims)
    '''
    def px(self, x: torch.Tensor, z: torch.Tensor) -> td.Independent:
        return self.decoder(x, z)

    @contextmanager
    def _freeze_params(sefl, module: torch.nn.Module):
        """Temporarily set requires_grad=False for all params in `module`."""
        flags = [p.requires_grad for p in module.parameters()]
        try:
            for p in module.parameters():
                p.requires_grad_(False)
            yield
        finally:
            for p, f in zip(module.parameters(), flags):
                p.requires_grad_(f)

    def variational_objective(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        qz_x = self._define_variational_family(x, **kwargs)   # MVN
        z = qz_x.rsample()                                    # reparameterized
        px_z = self.px(x, z)
        expected_log_lik = px_z.log_prob(x)                   # (B,)
        with self._freeze_params(self.prior):
            dkl = self.prior.kl_divergence(qz_x)
        elbo = expected_log_lik - dkl                         # (B,)

        # track
        #pdb.set_trace()
        self.elbo_tracker.update(elbo.mean())
        self.ell_tracker.update(expected_log_lik.mean())
        self.dkl_tracker.update(dkl.mean())

        loss = -elbo.mean()
        return loss, {"qz_x": qz_x, "expected_log_lik": expected_log_lik, "dkl": dkl, "elbo": elbo}

    # ----- one optimization step (pass optimizer explicitly) -----
    def variational_inference_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        loss, vf = self.variational_objective(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        optimizer.step()
        return loss, vf

    # ----- clustering helpers -----
    @torch.no_grad()
    def cluster_probabilities(self, z_samples: torch.Tensor):
        if hasattr(self.prior, "cluster_probabilities"):
            return self.prior.cluster_probabilities(samples=z_samples, encoder=lambda y, **kw: self._define_variational_family(y, **kw))
        return None

    @torch.no_grad()
    def additional_metrics(self, data: Dict[str, torch.Tensor], vf: Dict[str, Any]):
        # Plug in your own clustering_metrics(...) if available
        if "label" in data:
            z_samples = vf["qz_x"].rsample()         # (B, D)
            probs = self.cluster_probabilities(z_samples)
            if probs is not None:
                # Placeholder: set util=1 (to keep same API shape). Replace with real metrics.
                self.cluster_util.update(torch.tensor(1.0))
            else:
                self.cluster_util.update(torch.tensor(1.0))
        else:
            self.cluster_util.update(torch.tensor(1.0))

    # ----- Keras-style helpers (optional) -----
    @torch.no_grad()
    def train_step(self, data: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
        loss, vf = self.variational_inference_step(data["x"], optimizer)
        self.additional_metrics(data, vf)
        return self.get_metrics_result()

    @torch.no_grad()
    def test_step(self, data: Dict[str, torch.Tensor]):
        _loss, vf = self.variational_objective(data["x"], training=False)
        self.additional_metrics(data, vf)
        return self.get_metrics_result()

    @torch.no_grad()
    def predict_step(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        _loss, vf = self.variational_objective(data["x"], training=False)
        return vf["qz_x"].rsample()

    @torch.no_grad()
    def get_metrics_result(self) -> Dict[str, float]:
        return {
            "elbo": float(self.elbo_tracker.compute()),
            "ell":  float(self.ell_tracker.compute()),
            "dkl":  float(self.dkl_tracker.compute()),
            "clust": float(self.cluster_util.compute()),
            # add your own: "acc": ..., "ari": ..., "nmi": ...
        }


class EmpiricalBayesVariationalAutoencoder(VariationalAutoencoder, metaclass=abc.ABCMeta):
    def __init__(self, encoder: nn.Module, enc_out_dim: int, decoder: nn.Module, prior, **kwargs):
        super().__init__(encoder, enc_out_dim, decoder, prior, **kwargs)

        # --- split parameters: prior vs model (everything else) ---
        #self._prior_params  = list(self.prior.parameters())
        #_prior_ids = {id(p) for p in self._prior_params}
        #pdb.set_trace()
        #self._model_params  = [p for p in self.parameters() if p not in _prior_ids]

        # sanity: union equals all params (order can differ)
        #assert len(list(self.parameters())) == (len(self._model_params) + len(self._prior_params))

        # a dedicated optimizer for the **model** (user can replace this)
        #self.model_optimizer = optimizer

    @torch.no_grad()
    def _encoder_no_grad(self, x: torch.Tensor, **kwargs) -> td.Distribution:
        """
        Make an encoder callable for the prior that does NOT create a graph
        (so prior updates don't backprop into the encoder).
        """
        qz_x = self._define_variational_family(x, **kwargs)
        # Rebuild a distribution on detached parameters/tensors:
        '''
        if isinstance(qz_x, td.Independent) and isinstance(qz_x.base_dist, td.Normal):
            loc  = qz_x.base_dist.loc.detach()
            scale = qz_x.base_dist.scale.detach()
            return td.Independent(td.Normal(loc, scale), 1)
        else:
            # Fallback: sample and rewrap as diag normal with empirical mean/std (rare)
            z = qz_x.rsample().detach()
            mu = z
            std = torch.ones_like(z)
            return td.Independent(td.Normal(mu, std), 1)
        '''
        loc  = qz_x.base_dist.loc.detach()
        scale = qz_x.base_dist.scale.detach()
        return td.Independent(td.Normal(loc, scale), 1)

    @torch.no_grad()
    def get_metrics_result(self) -> Dict[str, float]:
        # reuse parent’s helper (already returns dict of floats)
        return super().get_metrics_result()

    @torch.no_grad()
    def test_step(self, data: Dict[str, torch.Tensor]):
        _, vf = self.variational_objective(data["x"], training=False)
        self.additional_metrics(data, vf)
        return self.get_metrics_result()

    def train_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Step 1: update model (encoder/decoder + likelihood scale) via ELBO.
        Step 2: update prior parameters via prior.inference_step (EB update).
        """
        # ----- Step 1: model update (no prior params in this optimizer) -----
        loss, vf = self.variational_inference_step(x, optimizer=optimizer)

        # ----- Step 2: prior-only update (prevent grads into encoder) -----
        # Provide a no-grad encoder callable so the prior updates only its params
        outputs = self.prior.inference_step(encoder=self._encoder_no_grad, x=x)
        outputs["vae-loss"] = loss

        # metrics
        #self.additional_metrics(data, vf)
        #return self.get_metrics_result()
        return outputs