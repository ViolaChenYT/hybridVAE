import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector-quantization bottleneck for VQ-VAE.

    Args
    ----
    n_e   : int
        Number of codebook vectors.
    e_dim : int
        Code dimensionality (must match encoder's output dim).
    beta  : float
        Commitment cost (beta * || z_e(x) - sg[e] ||^2).

    Notes
    -----
    - Accepts z of shape (B, D) [MLP encoder] or (B, C, H, W) [conv encoders].
    - Returns quantized z with the same shape as input.
    """

    def __init__(
        self, 
        n_e: int, 
        e_dim: int, 
        beta: float,
        #alpha: float = 1.0,
        embedding_init=None,
        trainable: bool = True,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(n_e, e_dim)
        if embedding_init is None:
            # Small, symmetric init (common for VQ)
            self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)
        else:
            w = torch.as_tensor(embedding_init, dtype=torch.float32)
            if w.shape != (n_e, e_dim):
                raise ValueError(f"Bad embedding_init shape {w.shape}, expected {(n_e, e_dim)}")
            self.embedding.weight.data = w.to(self.embedding.weight.device).clone()
        
        # freeze if requested
        self.embedding.weight.requires_grad = bool(trainable)

    def _flatten(self, z: torch.Tensor):
        """
        Flattens z to (N, D) where D == e_dim, and returns:
        z, z_flat, unflatten callable to restore original shape.
        """
        if z.dim() == 2:
            # (B, D)
            B, D = z.shape
            assert D == self.e_dim, f"z has dim {D}, expected {self.e_dim}"
            def unflatten(x):  # (B, D)
                return x
            return z, z, unflatten

        elif z.dim() == 4:
            # (B, C, H, W) -> (B*H*W, C)
            B, C, H, W = z.shape
            assert C == self.e_dim, f"z channel dim {C} != e_dim {self.e_dim}"
            z_nhwc = z.permute(0, 2, 3, 1).contiguous()   # (B,H,W,C)
            z_flat = z_nhwc.view(-1, self.e_dim)          # (B*H*W, C)

            def unflatten(x_flat):  # (B*H*W, C) -> (B,C,H,W)
                x = x_flat.view(B, H, W, self.e_dim).permute(0, 3, 1, 2).contiguous()
                return x
            return z, z_flat, unflatten

        else:
            raise ValueError(f"Unsupported z.dim()={z.dim()}. Expected 2 or 4.")

    def forward(self, z: torch.Tensor):
        """
        Returns
        -------
        loss : scalar tensor
            VQ loss = ||sg[z] - e||^2 + beta * ||z - sg[e]||^2
        z_q : tensor
            Quantized latents, same shape as z.
        perplexity : scalar tensor
            exp(H(code_usage)).
        one_hot : (N, n_e) one-hot encodings in flattened space.
        indices : (N,) argmin code indices in flattened space.
        """
        device = z.device

        # Flatten to (N, D)
        z_orig, z_flat, unflatten = self._flatten(z)       # z_flat: (N, D)

        # Compute squared distances to codebook: ||z - e||^2
        # (z^2) + (e^2) - 2 z·e
        e_w = self.embedding.weight                         # (n_e, D)
        z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)       # (N, 1)
        e_sq = (e_w ** 2).sum(dim=1)                        # (n_e,)
        # (N, n_e)
        distances = z_sq + e_sq.unsqueeze(0) - 2.0 * z_flat @ e_w.t()

        # Nearest code
        indices = torch.argmin(distances, dim=1)            # (N,)
        one_hot = F.one_hot(indices, num_classes=self.n_e).type(z.dtype)  # (N, n_e)

        # Quantize
        z_q_flat = one_hot @ e_w                             # (N, D)
        z_q = unflatten(z_q_flat)                            # back to z shape

        # VQ-VAE loss terms (embedding & commitment)
        #  stopgrad[z] -> codebook; stopgrad[e] -> encoder
        loss_embed = F.mse_loss(z_q.detach(), z_orig)        # ||sg[z] - e||^2
        loss_commit = F.mse_loss(z_q, z_orig.detach())       # ||z - sg[e]||^2
        loss = loss_embed + self.beta * loss_commit

        # Straight-through estimator
        z_q = z_orig + (z_q - z_orig).detach()

        # Perplexity
        avg_probs = one_hot.mean(dim=0)                      # (n_e,)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        return loss, z_q, perplexity, one_hot, indices


'''
### How to plug it into MLP setup
# Encoder produces (B, D) with D == e_dim
enc = Encoder(in_dim=..., h_dim=256, n_layers=2, out_dim=128)
vq  = VectorQuantizer(n_e=512, e_dim=128, beta=0.25)
dec = Decoder(z_dim=128, h_dim=256, n_layers=2, x_dim=..., out_activation='sigmoid')

x = ...                      # (B, x_dim) or (B, C, H, W) flattened before enc
z_e = enc(x)                 # (B, 128)
vq_loss, z_q, ppl, one_hot, idx = vq(z_e)
x_hat = dec(z_q)
'''