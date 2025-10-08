import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Simple MLP encoder.
    - in_dim:   input dimensionality AFTER flatten (e.g., C*H*W); if you pass images,
                set in_dim=None and it will infer from the first forward pass.
    - h_dim:    hidden width
    - n_layers: number of hidden layers
    - out_dim:  output size; for VQ-VAE this is the embedding dim d (pre-quantization)
    """
    def __init__(self, in_dim, h_dim, n_layers=2, out_dim=128, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim  # can be None to infer on first call
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.dropout = dropout

        self._build(in_dim)  # built lazily if in_dim is None
        '''
        if in_dim is not None:
            self._build(in_dim)
        '''

    def _build(self, in_dim):
        layers = [nn.Linear(in_dim, self.h_dim), nn.ReLU(inplace=True)]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        for _ in range(self.n_layers - 1):
            layers += [nn.Linear(self.h_dim, self.h_dim), nn.ReLU(inplace=True)]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        layers += [nn.Linear(self.h_dim, self.out_dim)]
        self._mlp = nn.Sequential(*layers)

    def forward(self, x):
        # flatten to (B, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        '''
        if self._mlp is None:
            # infer input dimension on first pass
            self._build(x.size(1))
        '''
        return self._mlp(x)