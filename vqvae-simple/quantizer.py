import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector-quantization bottleneck for VQ-VAE with EMA codebook updates + dead-code reset.

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

    Features
    --------
    - EMA codebook updates (use_ema=True)
    - FIFO buffer of recent z's for realistic reinit
    - Periodic reset of codes with low EMA usage

    Reset policy
    ------------
    At every `reset_interval` steps (forward calls in training mode),
    mark a code as dead if:
        ema_cluster_size[i] < usage_threshold
    and reinitialize its embedding vector by:
        - "samples": random recent z from buffer (preferred)
        - "gaussian": N(0, 1/sqrt(e_dim))
    Then synchronize EMA accumulators for that code to the new value.
    """

    def __init__(
        self, 
        n_e: int, 
        e_dim: int, 
        beta: float,
        #alpha: float = 1.0,
        embedding_init=None,
        trainable: bool = True,
        use_ema: bool = True, 
        decay: float = 0.99,
        eps: float = 1e-5,
        # --- reset options ---
        enable_reset: bool = False,
        reset_interval: int = 100,        # check every N training steps
        usage_threshold: float = 1.0,     # codes below this EMA count are "dead"
        reinit_strategy: str = "samples", # "samples" or "gaussian"
        buffer_size: int = 8192,          # recent z buffer
        **kwargs,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps

        self.enable_reset = enable_reset
        self.reset_interval = reset_interval
        self.usage_threshold = usage_threshold
        self.reinit_strategy = reinit_strategy
        self.buffer_size = buffer_size

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
        self.embedding.weight.requires_grad = bool(trainable and not use_ema)

        if self.use_ema:
            # EMA buffers: cluster sizes and sums of assigned z's
            self.register_buffer("ema_cluster_size", torch.zeros(n_e))
            self.register_buffer("ema_w", torch.zeros(n_e, e_dim))
            # Start EMA at current weights to avoid cold start
            self.ema_w.data.copy_(self.embedding.weight.data)
        
        # step counter + small FIFO buffer for z’s
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("buffer_z", torch.zeros(self.buffer_size, e_dim))
        self.register_buffer("buffer_filled", torch.zeros((), dtype=torch.long))  # how many valid rows


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
    
    @torch.no_grad()
    def _push_to_buffer(self, z_flat: torch.Tensor):
        """Push a random subset of z_flat into the FIFO buffer."""
        if self.buffer_size <=0:
            return 
        # sample up to buffer_size elements (or all if fewer)
        n = z_flat.shape[0]
        k = min(n, self.buffer_size // 4 if self.buffer_filled.item() > 0 else min(n, self.buffer_size))
        if k<=0:
            return 
        idx = torch.randperm(n, device=z_flat.device)[:k]
        samples = z_flat[idx].detach()
        # write into circular buffer
        bf = int(self.buffer_filled.item())
        if bf < self.buffer_size:
            # fill from start 
            end = min(self.buffer_size, bf + k)
            write = end - bf
            self.buffer_z[bf:end].copy_(samples[:write].to(self.buffer_z.device))
            self.buffer_filled.fill_(end)
            # if still remaining, wrap to start 
            rem = k - write
            if rem > 0:
                self.buffer_z[:rem].copy_(samples[write:k].to(self.buffer_z.device))
        else:
            # full: overwrite from a random start 
            start = torch.randint(0, self.buffer_size, (), device=z_flat.device).item()
            end = start + k
            if end <= self.buffer_size:
                self.buffer_z[start:end].copy_(samples.to(self.buffer_z.device))
            else:
                first = self.buffer_size - start
                self.buffer_z[start:].copy_(samples[:first].to(self.buffer_z.device))
                self.buffer_z[: end - self.buffer_size].copy_(samples[first:].to(self.buffer_z.device))
 
    @torch.no_grad()
    def _ema_update(self, z_flat: torch.Tensor, one_hot: torch.Tensor):
        # one_hot: (N, n_e), z_flat: (N, D)
        # Per-code counts and sums this batch
        cluster_size = one_hot.sum(dim=0) # (n_e,)
        embed_sums = one_hot.t() @ z_flat  # (n_e, D)

        # EMA updates 
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)
        self.ema_w.mul_(self.decay).add_(embed_sums, alpha=1.0 - self.decay)

        # Laplace-style smoothing to avoid dead codes, then renormalize
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.n_e * self.eps) * n  # (n_e,)

        # Normalize ema_w by counts to get new code vectors
        new_weight = self.ema_w / (smoothed.unsqueeze(1) + self.eps)
        self.embedding.weight.data.copy_(new_weight)

    @torch.no_grad()
    def _reset_dead_codes(self):
        '''
        if not self.enable_reset:
            return
        '''
        if (self.step.item() % self.reset_interval) != 0:
            return 
        
        dead = (self.ema_cluster_size < self.usage_threshold)
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return
        
        dead_idx = torch.nonzero(dead, as_tuple=False).squeeze(1)
        #  Reinit values
        if self.reinit_strategy == "samples" and self.buffer_filled.item() > 0:
            # sample with replacement from buffer
            bf = int(self.buffer_filled.item())
            choices = torch.randint(0, bf, (num_dead,), device=self.embedding.weight.device)
            new_vecs = self.buffer_z[choices].to(self.embedding.weight.device)
        else:
            # Gaussian fallback
            new_vecs = torch.randn(num_dead, self.e_dim, device=self.embedding.weight.device) / math.sqrt(self.e_dim)
        
        self.embedding.weight.data[dead_idx] = new_vecs
        if self.use_ema:
            # Set EMA sums to match the new vector times a tiny pseudo-count
            tiny = 1.0  # pseudo-count; keeps them alive for a while
            self.ema_cluster_size[dead_idx] = tiny
            self.ema_w[dead_idx] = new_vecs * tiny

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

        if self.use_ema:
            #loss_embed = z_q.detach().sub(z_orig).pow(2).mean() * 0.0  # no embed loss
            loss_commit = F.mse_loss(z_q, z_orig.detach())
            loss = self.beta * loss_commit
        else:
            # VQ-VAE loss terms (embedding & commitment)
            #  stopgrad[z] -> codebook; stopgrad[e] -> encoder
            loss_embed = F.mse_loss(z_q.detach(), z_orig)        # ||sg[z] - e||^2
            loss_commit = F.mse_loss(z_q, z_orig.detach())       # ||z - sg[e]||^2
            loss = loss_embed + self.beta * loss_commit

        # Straight-through estimator
        z_q = z_orig + (z_q - z_orig).detach()

        if self.training:
            with torch.no_grad():
                if self.use_ema:
                    self._ema_update(z_flat.detach(), one_hot.detach())
                if self.enable_reset:
                    self._push_to_buffer(z_flat.detach())
                    self.step.add_(1)
                    self._reset_dead_codes()
        
        '''
        if self.use_ema and self.training:
            with torch.no_grad():
                self._ema_update(z_flat.detach(), one_hot.detach())
        '''

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