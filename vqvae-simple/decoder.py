import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.distributions import NegativeBinomial

def softplus_inv(y: torch.Tensor) -> torch.Tensor:
    # numerically stable inverse of softplus
    return y + torch.log(-torch.expm1(-y))

class Decoder(nn.Module):
    """
    Simple MLP decoder p_phi(x|z).
    - z_dim:     latent size
    - h_dim:     hidden width
    - n_layers:  number of hidden layers (ReLU MLP)
    - x_dim:     flattened output dimension (ignored if out_shape is given)
    - out_shape: optional (C,H,W). If provided, output is reshaped to (B,C,H,W).
    - out_activation: None | 'sigmoid' | 'tanh'
        * 'sigmoid' for Bernoulli pixels in [0,1]
        * 'tanh' for outputs in [-1,1]
    """
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        n_layers: int,
        x_dim: Optional[int] = None,
        out_shape: Optional[Tuple[int,int,int]] = None,
        out_activation: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert (x_dim is not None) ^ (out_shape is not None), \
            "Specify exactly one of x_dim or out_shape."

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.out_shape = out_shape
        self.out_dim = x_dim if out_shape is None else int(torch.tensor(out_shape).prod().item())
        self.out_activation = out_activation

        layers = [nn.Linear(z_dim, h_dim), nn.ReLU(inplace=True)]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(h_dim, self.out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x_out = self.mlp(z)  # (B, out_dim)
        if self.out_activation == 'sigmoid':
            x_out = torch.sigmoid(x_out)
        elif self.out_activation == 'tanh':
            x_out = torch.tanh(x_out)

        if self.out_shape is not None:
            B = z.size(0)
            x_out = x_out.view(B, *self.out_shape)  # (B, C, H, W)
        return x_out

class NBDecoder(nn.Module):
    """
    p(x|z) = NB(mean=mu, inv_dispersion=theta)
    Parameterization: x ~ NB(theta, p), where mean mu = theta * (p/(1-p)).

    Options for theta:
      - per-sample, per-feature (default): theta_head(h)
      - per-feature shared across batch:  shared_theta=True
      - scalar shared across all features: scalar_theta=True
        (takes precedence over shared_theta if both are True)
    """
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        n_layers: int,
        x_dim: int,
        dropout: float = 0.0,
        shared_theta: bool = False,   # per-feature vector
        scalar_theta: bool = False,   # single scalar for all features
        theta_init: float = 0.5,      # initial positive value for learnable theta
        use_batchnorm: bool = True,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
        eps: float = 1e-8, 
        **kwargs,
    ):
        super().__init__()

        # ----- backbone (with optional BN) -----
        layers: list[nn.Module] = []
        in_dim = z_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, h_dim, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h_dim, eps=bn_eps, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.eps = eps

        # ----- heads / parameters -----
        self.mu_head = nn.Linear(h_dim, x_dim)
        nn.init.constant_(self.mu_head.bias, -2.0)

        self.theta_head = None
        self.theta_param = None
        self.theta_scalar = None

        if scalar_theta:
            # single learnable scalar (unconstrained param -> softplus in forward)
            init_u = softplus_inv(torch.tensor(float(theta_init)))
            self.theta_scalar = nn.Parameter(init_u)         # shape []
        elif shared_theta:
            # one learnable value per feature (shared across batch)
            init_u = softplus_inv(torch.full((1, x_dim), float(theta_init)))
            self.theta_param = nn.Parameter(init_u)          # shape [1, D]
        else:
            # per-sample, per-feature predicted from h
            self.theta_head = nn.Linear(h_dim, x_dim)
            nn.init.constant_(self.theta_head.bias, -2.0)

    def forward(self,  x: torch.Tensor, z: torch.Tensor):
        h = self.backbone(z)
        #mu = F.softplus(self.mu_head(h)).clamp_min(1e-8)  # [B, D]
        #library = 5000.0
        library = x.sum(dim=-1, keepdim=True)  # [B, 1]
        mu = F.softmax(self.mu_head(h), dim=-1) * library + self.eps  # [B, D]

        if self.theta_head is not None:
            theta = F.softplus(self.theta_head(h)).clamp_min(1e-8)         # [B, D]
        elif self.theta_param is not None:
            theta = F.softplus(self.theta_param).expand_as(mu).clamp_min(1e-8)  # [B, D]
        else:
            # scalar theta shared across all features & samples
            theta_scalar_pos = F.softplus(self.theta_scalar).clamp_min(1e-8)     # []
            theta = theta_scalar_pos.view(1, 1).expand_as(mu)                    # [B, D]

        return mu, theta  # both positive


def nb_nll_from_mu_theta(x_int, mu, theta, reduction="mean"):
    """
    Negative log-likelihood using torch.distributions.NegativeBinomial.
    Param mapping:
      total_count = theta
      logits = log(mu) - log(theta + mu)
      (equivalently, probs = theta / (theta + mu))
    """
    eps = 1e-8
    logits = torch.log(mu + eps) - torch.log(theta + mu + eps)
    dist = NegativeBinomial(total_count=theta, logits=logits)
    nll = -dist.log_prob(x_int)                  # shape (B, D)
    nll = nll.sum(-1)                            # sum over features -> (B,)
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll



class NBDecoder2(nn.Module):
    """
    p(x|z) = NB(mean=mu, inv_dispersion=theta)
    Parameterization: x ~ NB(theta, p), where mean mu = theta * (p/(1-p)).
    We output:
      mu    = softplus(W_mu h + b)
      theta = softplus(W_th h + b)  (can also be per-feature only)
    """
    def __init__(self, z_dim, h_dim, n_layers, x_dim, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(z_dim, h_dim), nn.ReLU(inplace=True)]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        self.px_scale_decoder = nn.Sequential(nn.Linear(h_dim, x_dim), nn.Softmax(dim=-1))
        self.px_r_decoder = nn.Linear(h_dim, x_dim)

    def forward(self, z, x):
        library = torch.sum(x, dim=1, keepdim=True).clamp_min(1e-8)
        h = self.backbone(z)
        px_scale = self.px_scale_decoder(h)
        px_rate = library * px_scale
        px_rate = px_rate.clamp(min=1e-8, max=1e12)
        px_r = torch.nn.functional.softplus(self.px_r_decoder(h)) + 1e-6
        px_r = px_r.clamp(min=1e-6, max=1e6) 
        return px_rate, px_r

class NBDecoder3(nn.Module):
    """
    μ = library * softmax(W_mu h)
    θ = softplus(θ_param)   # per-gene, not a function of z
    """
    def __init__(self, z_dim, h_dim, n_layers, x_dim, dropout=0.0, theta_init=1.0):
        super().__init__()
        layers = [nn.Linear(z_dim, h_dim), nn.ReLU(inplace=True)]
        if dropout > 0: layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout(dropout))
        self.backbone = nn.Sequential(*layers)

        self.px_scale_decoder = nn.Linear(h_dim, x_dim)
        # per-gene θ (inverse-dispersion), shared across cells
        self.theta_param = nn.Parameter(torch.full((1, x_dim), float(theta_init)))

    def forward(self, z, library):
        """
        z: (B, z_dim)
        library: (B, 1)  precomputed or predicted library size per cell
        Returns: mu (B,D), theta (B,D)
        """
        h = self.backbone(z)
        px_scale = F.softmax(self.px_scale_decoder(h), dim=-1)        # (B,D)
        mu = (library.clamp_min(1e-8) * px_scale).clamp(1e-8, 1e12)   # (B,D)

        theta = F.softplus(self.theta_param).expand_as(mu)            # (B,D)
        theta = theta.clamp(1e-6, 1e6)
        return mu, theta