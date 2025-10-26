import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Simple MLP encoder with optional BatchNorm1d.

    Args
    ----
    in_dim:    int | None   (None => infer at first forward)
    h_dim:     int
    n_layers:  int          (number of hidden layers)
    out_dim:   int
    dropout:   float
    use_batchnorm: bool     (enable BN between Linear and ReLU in hidden layers)
    bn_momentum: float
    bn_eps:    float
    """
    def __init__(
        self,
        in_dim,
        h_dim,
        n_layers: int = 2,
        out_dim: int = 128,
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps

        self._mlp = None
        if in_dim is not None:
            self._build(in_dim)

    def _build(self, in_dim: int):
        layers: list[nn.Module] = []
        cur = in_dim
        # Hidden layers
        for _ in range(self.n_layers):
            layers.append(nn.Linear(cur, self.h_dim, bias=not self.use_bn))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(self.h_dim, eps=self.bn_eps, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            cur = self.h_dim
        # Output layer (no BN here by default)
        layers.append(nn.Linear(cur, self.out_dim))
        self._mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten to (B, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        # Lazy build if in_dim was None
        if self._mlp is None:
            self._build(x.size(1))
        return self._mlp(x)