#!/usr/bin/env python3
"""
Comparison between theoretical optimal encoder/decoder and learned encoder/decoder
for the GMVAE with known generative process X = Az + epsilon.

Theoretical optimal:
- Encoder: A inverse (pseudo-inverse if not square)
- Decoder: A

This script compares the losses and performance between these approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
import math
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import the existing modules
from models import *
from priors_new import *

@torch.no_grad()
def generate_clustered_gauss_1d(
    n_clusters: int = 4,
    points_per_cluster: int = 300,
    n_features: int = 100,
    latent_dim: int = 1,
    sigma_val: float = 0.5,
    cluster_centers=None,
    cluster_spread: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
    *,
    nonlinear: bool = False,
    per_cluster_feats: int | None = None,   # if None, auto = (n_features - noise_feats)//n_clusters
    noise_feats: int = 20,
    bump_width: float | None = None,        # if None, auto = 2*cluster_spread
    freq_min: float = 0.8,
    freq_max: float = 2.5,
    uniform_noise_scale: float = 0.5,       # amplitude for background uniform noise features
):
    """
    Returns:
      X [N, D], labels [N], Z [N, latent_dim], centers [K, latent_dim], W, b
      Note: when nonlinear=True, W and b are returned as None.
    """
    assert latent_dim >= 1, "nonlinear mode uses z[:,0]"
    g = torch.Generator(device=device).manual_seed(seed)
    N = n_clusters * points_per_cluster

    # centers in latent space
    if cluster_centers is None:
        centers_1d = torch.linspace(-2, 2, n_clusters, device=device)
        centers = torch.zeros(n_clusters, latent_dim, device=device)
        centers[:, 0] = centers_1d
    else:
        centers = torch.as_tensor(cluster_centers, dtype=torch.float32, device=device)
        if centers.ndim == 1:
            centers = centers.unsqueeze(1)
        assert centers.shape == (n_clusters, latent_dim), \
            f"centers must be shape ({n_clusters},{latent_dim})"

    # assignments and latent samples
    labels = torch.arange(n_clusters, device=device).repeat_interleave(points_per_cluster)  # [N]
    z = torch.randn(N, latent_dim, generator=g, device=device) * cluster_spread
    z += centers[labels]

    if not nonlinear:
        # original linear map
        W = torch.randn(n_features, latent_dim, generator=g, device=device) / math.sqrt(latent_dim)
        b = torch.randn(n_features, generator=g, device=device) * 0.2
        mean_x = z @ W.T + b
        X = mean_x if sigma_val <= 0 else mean_x + sigma_val * torch.randn_like(mean_x, generator=g)
        return X, labels.to(device), z.to(device), centers.to(device), W.to(device), b.to(device)

    # ---- Nonlinear feature construction ----
    # auto-derive per-cluster feature count and bump width
    if per_cluster_feats is None:
        per_cluster_feats = max((n_features - noise_feats) // n_clusters, 0)
    tau = (2.0 * cluster_spread) if (bump_width is None) else float(bump_width)

    # reserve exactly n_features columns
    feats = []
    z1 = z[:, 0]  # [N]

    # per-cluster localized sin/cos features that peak near each center and decay elsewhere
    # feature form:  A * exp(-0.5 * ((z1 - c_k)/tau)^2) * sin_or_cos(omega * (z1 - c_k) + phase)
    for k in range(n_clusters):
        c_k = centers[k, 0]
        dx = z1 - c_k                                # [N]
        bump = torch.exp(-0.5 * (dx / tau) ** 2)     # [N]
        # sample frequencies per feature
        if per_cluster_feats > 0:
            omegas = torch.empty(per_cluster_feats, device=device).uniform_(freq_min, freq_max, generator=g)
            phases = torch.zeros(per_cluster_feats, device=device)
            # alternate sin/cos by toggling phase (0 -> cos at center; pi/2 -> sin shifted to peak at center)
            phases[1::2] = math.pi / 2.0

            A = 1.0  # amplitude
            for j in range(per_cluster_feats):
                y = A * bump * torch.cos(omegas[j] * dx + phases[j])
                feats.append(y.unsqueeze(1))         # [N,1]

    # background uniform noise features independent of z
    for _ in range(noise_feats):
        u = torch.empty(N, device=device).uniform_(-uniform_noise_scale, uniform_noise_scale, generator=g)
        feats.append(u.unsqueeze(1))

    # concatenate and fit to desired width
    if len(feats) == 0:
        Xnl = torch.zeros(N, 0, device=device)
    else:
        Xnl = torch.cat(feats, dim=1)  # [N, F_raw]

    # pad or trim to exactly n_features
    F_raw = Xnl.shape[1]
    if F_raw < n_features:
        pad = torch.zeros(N, n_features - F_raw, device=device)
        Xnl = torch.cat([Xnl, pad], dim=1)
    elif F_raw > n_features:
        Xnl = Xnl[:, :n_features]

    # additive observation noise
    if sigma_val > 0:
        Xnl = Xnl + sigma_val * torch.randn_like(Xnl)

    return Xnl, labels.to(device), z.to(device), centers.to(device), None, None


class TheoreticalOptimalModel(nn.Module):
    """
    Theoretical optimal encoder/decoder using the true generative process.
    Encoder: A inverse (pseudo-inverse)
    Decoder: A
    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor, sigma: float = 0.5):
        super().__init__()
        self.A = A  # Shape: (n_features, latent_dim)
        self.b = b  # Shape: (n_features,)
        self.sigma = sigma
        
        # Compute pseudo-inverse for encoder
        self.A_pinv = torch.pinverse(A)  # Shape: (latent_dim, n_features)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Theoretical optimal encoder: z = A^+ (x - b)
        Returns mean and log variance for diagonal Gaussian
        """
        # Remove bias: x_centered = x - b
        x_centered = x - self.b.unsqueeze(0)
        
        # Apply pseudo-inverse: z = A^+ x_centered
        z_mean = x_centered @ self.A_pinv.T
        
        # For theoretical optimal, we know the true latent, so we can compute
        # the theoretical variance. The encoder should be deterministic in the
        # noise-free case, but we add small variance for numerical stability
        z_logvar = torch.full_like(z_mean, math.log(self.sigma**2))
        
        return z_mean, z_logvar
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Theoretical optimal decoder: x = A z + b
        Returns mean and log variance
        """
        x_mean = z @ self.A.T + self.b.unsqueeze(0)
        x_logvar = torch.full_like(x_mean, math.log(self.sigma**2))
        
        return x_mean, x_logvar
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode -> decode
        Returns: z_mean, z_logvar, x_mean, x_logvar
        """
        z_mean, z_logvar = self.encode(x)
        x_mean, x_logvar = self.decode(z_mean)
        return z_mean, z_logvar, x_mean, x_logvar


class TheoreticalVAE(nn.Module):
    """
    VAE using theoretical optimal encoder/decoder with proper probabilistic formulation
    """
    def __init__(self, A: torch.Tensor, b: torch.Tensor, sigma: float = 0.5, prior=None):
        super().__init__()
        self.A = A
        self.b = b
        self.sigma = sigma
        self.prior = prior
        self.A_pinv = torch.pinverse(A)
        
    def _define_variational_family(self, x: torch.Tensor):
        """Theoretical optimal q(z|x)"""
        x_centered = x - self.b.unsqueeze(0)
        z_mean = x_centered @ self.A_pinv.T
        
        # For theoretical optimal, the variance should be related to the noise
        # In the noise-free case, this should be deterministic
        z_std = torch.full_like(z_mean, self.sigma)
        
        return torch.distributions.Independent(
            torch.distributions.Normal(z_mean, z_std), 1
        )
    
    def px(self, z: torch.Tensor, x_shape: torch.Size):
        """Theoretical optimal p(x|z)"""
        x_mean = z @ self.A.T + self.b.unsqueeze(0)
        x_std = torch.full_like(x_mean, self.sigma)
        
        base = torch.distributions.Normal(loc=x_mean, scale=x_std)
        return torch.distributions.Independent(base, x_mean.dim() - 1)
    
    def variational_objective(self, x: torch.Tensor):
        """Compute ELBO for theoretical model"""
        qz_x = self._define_variational_family(x)
        z = qz_x.rsample()
        px_z = self.px(z, x.shape)
        
        # Expected log likelihood
        expected_log_lik = px_z.log_prob(x)
        
        # KL divergence (use standard normal prior if no prior specified)
        if self.prior is not None:
            dkl = self.prior.kl_divergence(qz_x)
        else:
            # Standard normal prior
            pz = torch.distributions.Independent(
                torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z)), 1
            )
            dkl = torch.distributions.kl_divergence(qz_x, pz)
        
        elbo = expected_log_lik - dkl
        loss = -elbo.mean()
        
        return loss, {"qz_x": qz_x, "elbo": elbo, "expected_log_lik": expected_log_lik, "dkl": dkl}


def compare_models(X, y, Z_true, W, b, sigma, device):
    """
    Compare theoretical optimal model with learned model
    """
    print("="*80)
    print("THEORETICAL vs LEARNED ENCODER/DECODER COMPARISON")
    print("="*80)
    
    # 1. Theoretical Optimal Model
    print("\n1. THEORETICAL OPTIMAL MODEL")
    print("-" * 40)
    
    theoretical_model = TheoreticalOptimalModel(W, b, sigma)
    theoretical_vae = TheoreticalVAE(W, b, sigma)
    
    # Test theoretical model
    with torch.no_grad():
        z_mean, z_logvar, x_mean, x_logvar = theoretical_model(X)
        
        # Compute reconstruction error
        recon_error = F.mse_loss(x_mean, X)
        print(f"Theoretical reconstruction error (MSE): {recon_error.item():.6f}")
        
        # Compute theoretical latent correlation with true latent
        z_corr = torch.corrcoef(torch.stack([Z_true.flatten(), z_mean.flatten()]))[0, 1]
        print(f"Theoretical latent correlation with true: {z_corr.item():.6f}")
        
        # Compute ELBO for theoretical model
        theoretical_loss, theoretical_outputs = theoretical_vae.variational_objective(X)
        print(f"Theoretical ELBO loss: {theoretical_loss.item():.6f}")
        print(f"Theoretical expected log likelihood: {theoretical_outputs['expected_log_lik'].mean().item():.6f}")
        print(f"Theoretical KL divergence: {theoretical_outputs['dkl'].mean().item():.6f}")
    
    # 2. Learned Model (from the notebook)
    print("\n2. LEARNED MODEL")
    print("-" * 40)
    
    # Build learned model (same as in notebook)
    model_prior = GaussianMixture(latent_dim=1, num_clusters=4)
    model_encoder = build_encoder(dim_x=100, h_dim=64, n_layers=2)
    model_decoder = build_decoder(dim_x=100, latent_dim=1, h_dim=64, n_layers=2)
    # learned_model = VariationalAutoencoder(
    #     encoder=model_encoder, 
    #     enc_out_dim=64, 
    #     decoder=model_decoder, 
    #     prior=model_prior
    # )
    learned_model = EmpiricalBayesVariationalAutoencoder(encoder=model_encoder, enc_out_dim=64, decoder=model_decoder, prior=model_prior)

    
    learned_model.to(device)
    learned_model.prior.to(device)
    
    # Train the learned model (simplified training)
    print("Training learned model...")
    optimizer = torch.optim.Adam(learned_model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    learned_model.train()
    for epoch in range(100):  # Reduced epochs for comparison
        for batch_x in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X), batch_size=128, shuffle=True):
            batch_x = batch_x[0].to(device)
            loss, _ = learned_model.variational_inference_step(batch_x, optimizer)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # Evaluate learned model
    learned_model.eval()
    with torch.no_grad():
        learned_qz_x = learned_model._define_variational_family(X)
        learned_z_mean = learned_qz_x.mean
        learned_z_sample = learned_qz_x.sample()
        
        # Compute reconstruction
        learned_recon = learned_model.decoder(learned_z_sample)
        if isinstance(learned_recon, tuple):
            learned_x_mean, learned_x_logvar = learned_recon
        else:
            # Handle case where decoder returns single tensor
            learned_x_mean = learned_recon
        
        learned_recon_error = F.mse_loss(learned_x_mean, X)
        print(f"Learned reconstruction error (MSE): {learned_recon_error.item():.6f}")
        
        # Compute learned latent correlation with true latent
        learned_z_corr = torch.corrcoef(torch.stack([Z_true.flatten(), learned_z_mean.flatten()]))[0, 1]
        print(f"Learned latent correlation with true: {learned_z_corr.item():.6f}")
        
        # Compute ELBO for learned model
        learned_loss, learned_outputs = learned_model.variational_objective(X)
        print(f"Learned ELBO loss: {learned_loss.item():.6f}")
        print(f"Learned expected log likelihood: {learned_outputs['expected_log_lik'].mean().item():.6f}")
        print(f"Learned KL divergence: {learned_outputs['dkl'].mean().item():.6f}")
    
    # 3. Comparison and Analysis
    print("\n3. COMPARISON ANALYSIS")
    print("-" * 40)
    
    print(f"Reconstruction Error:")
    print(f"  Theoretical: {recon_error.item():.6f}")
    print(f"  Learned:    {learned_recon_error.item():.6f}")
    print(f"  Ratio (Learned/Theoretical): {learned_recon_error.item() / recon_error.item():.2f}")
    
    print(f"\nLatent Correlation with True:")
    print(f"  Theoretical: {z_corr.item():.6f}")
    print(f"  Learned:    {learned_z_corr.item():.6f}")
    
    print(f"\nELBO Loss:")
    print(f"  Theoretical: {theoretical_loss.item():.6f}")
    print(f"  Learned:    {learned_loss.item():.6f}")
    print(f"  Ratio (Learned/Theoretical): {learned_loss.item() / theoretical_loss.item():.2f}")
    
    print(f"\nExpected Log Likelihood:")
    print(f"  Theoretical: {theoretical_outputs['expected_log_lik'].mean().item():.6f}")
    if 'expected_log_lik' in learned_outputs:
        print(f"  Learned:    {learned_outputs['expected_log_lik'].mean().item():.6f}")
    else:
        print(f"  Learned:    N/A")
    
    print(f"\nKL Divergence:")
    print(f"  Theoretical: {theoretical_outputs['dkl'].mean().item():.6f}")
    if 'dkl' in learned_outputs:
        print(f"  Learned:    {learned_outputs['dkl'].mean().item():.6f}")
    else:
        print(f"  Learned:    N/A")
    
    # 4. Visualization
    print("\n4. VISUALIZATION")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # True latent space
    axes[0, 0].scatter(Z_true.cpu().numpy(), np.zeros_like(Z_true.cpu().numpy()), 
                      c=y.cpu().numpy(), cmap='tab10', alpha=0.7, s=20)
    axes[0, 0].set_title('True Latent Space')
    axes[0, 0].set_xlabel('Latent Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Theoretical latent space
    axes[0, 1].scatter(z_mean.cpu().numpy(), np.zeros_like(z_mean.cpu().numpy()), 
                      c=y.cpu().numpy(), cmap='tab10', alpha=0.7, s=20)
    axes[0, 1].set_title('Theoretical Optimal Latent')
    axes[0, 1].set_xlabel('Latent Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learned latent space
    axes[0, 2].scatter(learned_z_mean.cpu().numpy(), np.zeros_like(learned_z_mean.cpu().numpy()), 
                      c=y.cpu().numpy(), cmap='tab10', alpha=0.7, s=20)
    axes[0, 2].set_title('Learned Latent')
    axes[0, 2].set_xlabel('Latent Value')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Reconstruction comparison
    sample_indices = np.random.choice(len(X), 100, replace=False)
    x_sample = X[sample_indices].cpu().numpy()
    theoretical_recon = x_mean[sample_indices].cpu().numpy()
    learned_recon = learned_x_mean[sample_indices].cpu().numpy()
    
    axes[1, 0].scatter(x_sample.flatten(), theoretical_recon.flatten(), alpha=0.6, s=10)
    axes[1, 0].plot([x_sample.min(), x_sample.max()], [x_sample.min(), x_sample.max()], 'r--', alpha=0.8)
    axes[1, 0].set_title('Theoretical Reconstruction')
    axes[1, 0].set_xlabel('True X')
    axes[1, 0].set_ylabel('Reconstructed X')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].scatter(x_sample.flatten(), learned_recon.flatten(), alpha=0.6, s=10)
    axes[1, 1].plot([x_sample.min(), x_sample.max()], [x_sample.min(), x_sample.max()], 'r--', alpha=0.8)
    axes[1, 1].set_title('Learned Reconstruction')
    axes[1, 1].set_xlabel('True X')
    axes[1, 1].set_ylabel('Reconstructed X')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Latent correlation comparison
    Z_true_cpu = Z_true.cpu().numpy()
    z_mean_cpu = z_mean.cpu().numpy()
    learned_z_mean_cpu = learned_z_mean.cpu().numpy()
    
    axes[1, 2].scatter(Z_true_cpu, z_mean_cpu, alpha=0.6, s=10, label='Theoretical')
    axes[1, 2].scatter(Z_true_cpu, learned_z_mean_cpu, alpha=0.6, s=10, label='Learned')
    axes[1, 2].plot([Z_true_cpu.min(), Z_true_cpu.max()], [Z_true_cpu.min(), Z_true_cpu.max()], 'r--', alpha=0.8)
    axes[1, 2].set_title('Latent Correlation')
    axes[1, 2].set_xlabel('True Latent')
    axes[1, 2].set_ylabel('Predicted Latent')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'theoretical': {
            'recon_error': recon_error.item(),
            'latent_corr': z_corr.item(),
            'elbo_loss': theoretical_loss.item(),
            'expected_log_lik': theoretical_outputs['expected_log_lik'].mean().item(),
            'dkl': theoretical_outputs['dkl'].mean().item()
        },
        'learned': {
            'recon_error': learned_recon_error.item(),
            'latent_corr': learned_z_corr.item(),
            'elbo_loss': learned_loss.item(),
            'expected_log_lik': learned_outputs['expected_log_lik'].mean().item(),
            'dkl': learned_outputs['dkl'].mean().item()
        }
    }


def main():
    """Main function to run the comparison"""
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    torch.manual_seed(42)
    
    # Generate the same dataset as in the notebook
    n_clusters = 4
    points_per_cluster = 300
    n_features = 100
    latent_dim = 1
    sigma_val = 0.5
    cluster_centers = [-2.0, -0.75, 0.75, 2.0]
    
    X, y, Z_true, true_centers, W, b = generate_clustered_gauss_1d(
        n_clusters=n_clusters,
        points_per_cluster=points_per_cluster,
        n_features=n_features,
        latent_dim=latent_dim,
        sigma_val=sigma_val,
        cluster_centers=cluster_centers,
        cluster_spread=0.2,
        seed=42,
        device=device,
    )
    
    print(f"Generated dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Z_true shape: {Z_true.shape}")
    print(f"  True cluster centers: {true_centers.flatten().tolist()}")
    print(f"  A matrix shape: {W.shape}")
    print(f"  b vector shape: {b.shape}")
    
    # Run comparison
    results = compare_models(X, y, Z_true, W, b, sigma_val, device)
    
    # Save results
    print(f"\n5. SUMMARY")
    print("-" * 40)
    print("Theoretical model should perform better since it uses the true generative process.")
    print("This comparison helps understand how much the learned model deviates from optimal.")
    
    return results


if __name__ == "__main__":
    results = main()
