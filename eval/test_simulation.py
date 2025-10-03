#!/usr/bin/env python3
"""
Simulation and Testing Script for VAE and GMVAE Models

This script provides comprehensive simulation and testing capabilities for:
1. Generating synthetic clustered datasets with known ground truth
2. Training and comparing VAE and GMVAE models
3. Evaluating clustering performance and latent space recovery
4. Comprehensive visualization and analysis

Usage:
    python test_simulation.py --model_type gmvae --n_clusters 4 --n_epochs 500
    python test_simulation.py --model_type vae --n_clusters 5 --latent_dim 2
    python test_simulation.py --model_type both --compare_models
"""

import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
from torch.utils.data import DataLoader, TensorDataset
import math
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

# Add src directory to path
module_path = os.path.abspath(os.path.join('.', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

from gmvae import GMVAE
from vae import VAE
from circular_vae import CircularVAE
from trainer import Trainer, set_seed

# set_seed function is now imported from trainer module

@torch.no_grad()
def generate_clustered_nb_1d(
    n_clusters=4,
    points_per_cluster=300,
    n_features=100,
    latent_dim=1,
    theta_val=12.0,
    cluster_centers=None,
    cluster_spread=0.2,
    seed=42,
    device="cpu",
):
    """
    Generate clustered dataset with 1D latent space.
    
    Args:
        n_clusters: Number of clusters
        points_per_cluster: Points per cluster
        n_features: Number of features
        latent_dim: Latent dimension (should be 1 for this function)
        theta_val: Negative binomial dispersion parameter
        cluster_centers: List of cluster centers (if None, auto-generated)
        cluster_spread: Standard deviation of cluster points
        seed: Random seed
        device: Device to use
    
    Returns:
        X: Count data tensor [N, D]
        labels: Cluster labels [N]
        Z_true: True latent values [N, 1]
        true_centers: True cluster centers [K, 1]
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N = n_clusters * points_per_cluster

    # Set cluster centers in 1D
    if cluster_centers is None:
        centers = torch.linspace(-2, 2, n_clusters, device=device).unsqueeze(1)
    else:
        centers = torch.tensor(cluster_centers, dtype=torch.float32, device=device).unsqueeze(1)

    # Sample cluster assignments
    labels = torch.arange(n_clusters, device=device).repeat_interleave(points_per_cluster)
    # True latent z
    z = torch.randn(N, latent_dim, generator=g, device=device) * cluster_spread
    z += centers[labels]

    # Decoder-ish linear map to gene space
    W = torch.randn(n_features, latent_dim, generator=g, device=device) / math.sqrt(latent_dim)
    b = torch.randn(n_features, generator=g, device=device) * 0.2

    logits = z @ W.T + b
    px_scale = F.softmax(logits, dim=-1)
    # Log-normal library sizes around ~5k counts
    lib = torch.exp(torch.randn(N, 1, generator=g, device=device) * 0.25 + 8.5)
    mu = lib * px_scale

    theta = torch.full_like(mu, float(theta_val))
    p = theta / (theta + mu + 1e-8)
    nb = NegativeBinomial(total_count=theta, probs=p)
    X = nb.sample()

    return X, labels.cpu(), z.cpu(), centers.cpu()

@torch.no_grad()
def generate_clustered_nb_2d(
    n_clusters=4,
    points_per_cluster=300,
    n_features=60,
    latent_dim=2,
    theta_val=12.0,
    radius=3.0,
    cluster_spread=0.25,
    seed=123,
    device="cpu",
):
    """
    Generate clustered dataset with 2D latent space (circular arrangement).
    
    Args:
        n_clusters: Number of clusters
        points_per_cluster: Points per cluster
        n_features: Number of features
        latent_dim: Latent dimension (should be 2 for this function)
        theta_val: Negative binomial dispersion parameter
        radius: Radius for circular cluster arrangement
        cluster_spread: Standard deviation of cluster points
        seed: Random seed
        device: Device to use
    
    Returns:
        X: Count data tensor [N, D]
        labels: Cluster labels [N]
        Z_true: True latent values [N, 2]
    """
    g = torch.Generator(device=device).manual_seed(seed)
    N = n_clusters * points_per_cluster

    # Arrange cluster centers on a circle
    centers = []
    for k in range(n_clusters):
        angle = 2 * math.pi * k / n_clusters
        centers.append([radius * math.cos(angle), radius * math.sin(angle)])
    centers = torch.tensor(centers, dtype=torch.float32, device=device)

    # Sample cluster assignments
    labels = torch.arange(n_clusters, device=device).repeat_interleave(points_per_cluster)
    # True latent z
    z = torch.randn(N, latent_dim, generator=g, device=device) * cluster_spread
    z += centers[labels]

    # Decoder-ish linear map to gene space
    W = torch.randn(n_features, latent_dim, generator=g, device=device) / math.sqrt(latent_dim)
    b = torch.randn(n_features, generator=g, device=device) * 0.2

    logits = z @ W.T + b
    px_scale = F.softmax(logits, dim=-1)
    # Log-normal library sizes around ~5k counts
    lib = torch.exp(torch.randn(N, 1, generator=g, device=device) * 0.25 + 8.5)
    mu = lib * px_scale

    theta = torch.full_like(mu, float(theta_val))
    p = theta / (theta + mu + 1e-8)
    nb = NegativeBinomial(total_count=theta, probs=p)
    X = nb.sample()

    return X, labels.cpu(), z.cpu()

# train_model function is now replaced by Trainer class

# evaluate_model function is now replaced by Trainer.evaluate method

def plot_training_losses(gmvae_losses=None, vae_losses=None, save_path=None):
    """Plot training losses."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total loss
    ax = axes[0]
    if gmvae_losses:
        gmvae_total = [h['loss'] for h in gmvae_losses]
        ax.plot(gmvae_total, label='GMVAE', alpha=0.8)
    if vae_losses:
        vae_total = [h['loss'] for h in vae_losses]
        ax.plot(vae_total, label='VAE', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Reconstruction loss
    ax = axes[1]
    if gmvae_losses:
        gmvae_recon = [h.get('recon_loss', 0) for h in gmvae_losses]
        ax.plot(gmvae_recon, label='GMVAE', alpha=0.8)
    if vae_losses:
        vae_recon = [h.get('recon_loss', 0) for h in vae_losses]
        ax.plot(vae_recon, label='VAE', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Reconstruction Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL loss
    ax = axes[2]
    if gmvae_losses:
        gmvae_kl = [h.get('kl_local', 0) for h in gmvae_losses]
        ax.plot(gmvae_kl, label='GMVAE', alpha=0.8)
    if vae_losses:
        vae_kl = [h.get('kl_local', 0) for h in vae_losses]
        ax.plot(vae_kl, label='VAE', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.set_title('KL Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_latent_space_comparison(Z_true, gmvae_metrics=None, vae_metrics=None, labels=None, 
                                latent_dim=1, save_path=None):
    """Plot comprehensive latent space comparison."""
    n_plots = 2 + (1 if gmvae_metrics else 0) + (1 if vae_metrics else 0)
    if latent_dim == 1:
        fig, axes = plt.subplots(2, n_plots, figsize=(4*n_plots, 8))
    else:
        fig, axes = plt.subplots(2, n_plots, figsize=(4*n_plots, 8))
    
    plot_idx = 0
    
    # True latent space
    ax = axes[0, plot_idx] if latent_dim == 1 else axes[0, plot_idx]
    if latent_dim == 1:
        for i in range(len(np.unique(labels))):
            mask = labels == i
            ax.scatter(Z_true[mask, 0], np.zeros_like(Z_true[mask, 0]), 
                      c=[plt.cm.tab10(i)], label=f'Cluster {i}', alpha=0.7, s=20)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_ylabel('')
        ax.set_yticks([])
    else:
        ax.scatter(Z_true[:, 0], Z_true[:, 1], c=labels, s=8, cmap="tab10")
        ax.set_ylabel('z2')
    ax.set_xlabel('True Latent Value' if latent_dim == 1 else 'z1')
    ax.set_title('True Latent Space')
    ax.legend() if latent_dim == 1 else None
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # GMVAE latent space
    if gmvae_metrics:
        ax = axes[0, plot_idx] if latent_dim == 1 else axes[0, plot_idx]
        enc_mu = gmvae_metrics['enc_mu']
        if latent_dim == 1:
            for i in range(len(np.unique(labels))):
                mask = labels == i
                ax.scatter(enc_mu[mask, 0], np.zeros_like(enc_mu[mask, 0]), 
                          c=[plt.cm.tab10(i)], label=f'Cluster {i}', alpha=0.7, s=20)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('')
            ax.set_yticks([])
        else:
            ax.scatter(enc_mu[:, 0], enc_mu[:, 1], c=labels, s=8, cmap="tab10")
            ax.set_ylabel('μ2')
        ax.set_xlabel('GMVAE Encoded Latent' if latent_dim == 1 else 'μ1')
        ax.set_title(f'GMVAE Latent (r={gmvae_metrics["correlation"]:.3f})')
        ax.legend() if latent_dim == 1 else None
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # VAE latent space
    if vae_metrics:
        ax = axes[0, plot_idx] if latent_dim == 1 else axes[0, plot_idx]
        enc_mu = vae_metrics['enc_mu']
        if latent_dim == 1:
            for i in range(len(np.unique(labels))):
                mask = labels == i
                ax.scatter(enc_mu[mask, 0], np.zeros_like(enc_mu[mask, 0]), 
                          c=[plt.cm.tab10(i)], label=f'Cluster {i}', alpha=0.7, s=20)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.set_ylabel('')
            ax.set_yticks([])
        else:
            ax.scatter(enc_mu[:, 0], enc_mu[:, 1], c=labels, s=8, cmap="tab10")
            ax.set_ylabel('μ2')
        ax.set_xlabel('VAE Encoded Latent' if latent_dim == 1 else 'μ1')
        ax.set_title(f'VAE Latent (r={vae_metrics["correlation"]:.3f})')
        ax.legend() if latent_dim == 1 else None
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Correlation plots
    plot_idx = 0
    
    # GMVAE vs True correlation
    if gmvae_metrics:
        ax = axes[1, plot_idx] if latent_dim == 1 else axes[1, plot_idx]
        colors = [plt.cm.tab10(i) for i in labels.numpy()]
        ax.scatter(Z_true.numpy().flatten(), gmvae_metrics['enc_mu'].numpy().flatten(), 
                  c=colors, alpha=0.6, s=20)
        ax.plot([-3, 3], [-3, 3], 'r--', alpha=0.8, label='Perfect correlation')
        ax.set_xlabel('True Latent Value')
        ax.set_ylabel('GMVAE Encoded Value')
        ax.set_title(f'GMVAE vs True (r={gmvae_metrics["correlation"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # VAE vs True correlation
    if vae_metrics:
        ax = axes[1, plot_idx] if latent_dim == 1 else axes[1, plot_idx]
        colors = [plt.cm.tab10(i) for i in labels.numpy()]
        ax.scatter(Z_true.numpy().flatten(), vae_metrics['enc_mu'].numpy().flatten(), 
                  c=colors, alpha=0.6, s=20)
        ax.plot([-3, 3], [-3, 3], 'r--', alpha=0.8, label='Perfect correlation')
        ax.set_xlabel('True Latent Value')
        ax.set_ylabel('VAE Encoded Value')
        ax.set_title(f'VAE vs True (r={vae_metrics["correlation"]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Residuals plots
    if gmvae_metrics:
        ax = axes[1, plot_idx] if latent_dim == 1 else axes[1, plot_idx]
        residuals = gmvae_metrics['enc_mu'].numpy().flatten() - Z_true.numpy().flatten()
        colors = [plt.cm.tab10(i) for i in labels.numpy()]
        ax.scatter(Z_true.numpy().flatten(), residuals, c=colors, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('True Latent Value')
        ax.set_ylabel('GMVAE Residuals')
        ax.set_title(f'GMVAE Residuals (RMSE={gmvae_metrics["rmse"]:.3f})')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    if vae_metrics:
        ax = axes[1, plot_idx] if latent_dim == 1 else axes[1, plot_idx]
        residuals = vae_metrics['enc_mu'].numpy().flatten() - Z_true.numpy().flatten()
        colors = [plt.cm.tab10(i) for i in labels.numpy()]
        ax.scatter(Z_true.numpy().flatten(), residuals, c=colors, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('True Latent Value')
        ax.set_ylabel('VAE Residuals')
        ax.set_title(f'VAE Residuals (RMSE={vae_metrics["rmse"]:.3f})')
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_encoded_latent_distribution(metrics, labels, model_name="Model", save_path=None):
    """Plot encoded latent distribution by true cluster labels."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    enc_mu = metrics['enc_mu']
    unique_labels = np.unique(labels)
    
    # Plot density histograms for each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        cluster_data = enc_mu[mask, 0].numpy()
        ax.hist(cluster_data, bins=50, density=True, alpha=0.6, 
                label=f'True Cluster {label}', color=plt.cm.tab10(i))
    
    ax.set_xlabel('Encoded Latent Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Encoded Latent Distribution - {model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_clustering_analysis(gmvae_metrics, labels, save_path=None):
    """Plot clustering-specific analysis for GMVAE."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion matrix
    ax = axes[0, 0]
    cm = gmvae_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Cluster')
    ax.set_title('Confusion Matrix')
    
    # Cluster assignment confidence
    ax = axes[0, 1]
    max_probs = torch.max(gmvae_metrics['cluster_probs'], dim=1)[0]
    ax.hist(max_probs.numpy(), bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Max Cluster Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Cluster Assignment Confidence')
    ax.grid(True, alpha=0.3)
    
    # Predicted vs True clusters
    ax = axes[1, 0]
    predicted = gmvae_metrics['predicted_clusters'].numpy()
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(predicted[mask], np.zeros_like(predicted[mask]), 
                  c=[plt.cm.tab10(i)], label=f'True Cluster {label}', alpha=0.7, s=20)
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_title('Predicted Clusters (colored by true)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance metrics
    ax = axes[1, 1]
    metrics = ['ARI', 'NMI', 'Confidence']
    values = [gmvae_metrics['ari'], gmvae_metrics['nmi'], gmvae_metrics['mean_confidence']]
    bars = ax.bar(metrics, values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
    ax.set_ylabel('Value')
    ax.set_title('Clustering Performance')
    ax.set_ylim(0, 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_summary(gmvae_metrics=None, vae_metrics=None, args=None):
    """Print comprehensive comparison summary."""
    print("="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    if args:
        print(f"Dataset Configuration:")
        print(f"  - Model type: {args.model_type}")
        print(f"  - Clusters: {args.n_clusters}")
        print(f"  - Latent dimension: {args.latent_dim}")
        print(f"  - Features: {args.n_features}")
        print(f"  - Points per cluster: {args.points_per_cluster}")
        print(f"  - Total samples: {args.n_clusters * args.points_per_cluster}")
        print()
    
    print(f"Training Configuration:")
    print(f"  - Epochs: {args.n_epochs if args else 'N/A'}")
    print(f"  - Learning rate: {args.learning_rate if args else 'N/A'}")
    print(f"  - Batch size: {args.batch_size if args else 'N/A'}")
    print()
    
    if gmvae_metrics:
        print(f"GMVAE Performance:")
        print(f"  - Latent space correlation: {gmvae_metrics['correlation']:.4f}")
        print(f"  - RMSE: {gmvae_metrics['rmse']:.4f}")
        print(f"  - ARI: {gmvae_metrics['ari']:.4f}")
        print(f"  - NMI: {gmvae_metrics['nmi']:.4f}")
        print(f"  - Mean cluster confidence: {gmvae_metrics['mean_confidence']:.4f}")
        print()
    
    if vae_metrics:
        print(f"VAE Performance:")
        print(f"  - Latent space correlation: {vae_metrics['correlation']:.4f}")
        print(f"  - RMSE: {vae_metrics['rmse']:.4f}")
        print()
    
    if gmvae_metrics and vae_metrics:
        print(f"Comparison:")
        if gmvae_metrics['correlation'] > vae_metrics['correlation']:
            print("  ✓ GMVAE shows BETTER latent space recovery than VAE")
        else:
            print("  ✗ VAE shows BETTER latent space recovery than GMVAE")
        
        if gmvae_metrics['rmse'] < vae_metrics['rmse']:
            print("  ✓ GMVAE shows LOWER reconstruction error than VAE")
        else:
            print("  ✗ VAE shows LOWER reconstruction error than GMVAE")
        
        if gmvae_metrics['ari'] > 0.8:
            print("  ✓ GMVAE shows EXCELLENT clustering performance")
        elif gmvae_metrics['ari'] > 0.6:
            print("  ~ GMVAE shows GOOD clustering performance")
        elif gmvae_metrics['ari'] > 0.4:
            print("  ~ GMVAE shows MODERATE clustering performance")
        else:
            print("  ✗ GMVAE shows POOR clustering performance")
    
    print("="*80)

def run_simulation(args):
    """Main simulation function."""
    print("="*80)
    print("VAE/GMVAE SIMULATION AND TESTING")
    print("="*80)
    
    # Set device and seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    set_seed(args.seed)
    
    # Generate synthetic data
    print(f"\n=== Generating Synthetic Data ===")
    if args.latent_dim == 1:
        cluster_centers = args.cluster_centers if args.cluster_centers else None
        X, labels, Z_true, true_centers = generate_clustered_nb_1d(
            n_clusters=args.n_clusters,
            points_per_cluster=args.points_per_cluster,
            n_features=args.n_features,
            latent_dim=args.latent_dim,
            theta_val=args.theta_val,
            cluster_centers=cluster_centers,
            cluster_spread=args.cluster_spread,
            seed=args.seed,
            device=device
        )
        print(f"Generated 1D dataset:")
        print(f"  X shape: {X.shape}")
        print(f"  True cluster centers: {true_centers.flatten().tolist()}")
    else:
        X, labels, Z_true = generate_clustered_nb_2d(
            n_clusters=args.n_clusters,
            points_per_cluster=args.points_per_cluster,
            n_features=args.n_features,
            latent_dim=args.latent_dim,
            theta_val=args.theta_val,
            radius=args.radius,
            cluster_spread=args.cluster_spread,
            seed=args.seed,
            device=device
        )
        print(f"Generated 2D dataset:")
        print(f"  X shape: {X.shape}")
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  Z_true shape: {Z_true.shape}")
    print(f"  Number of clusters: {args.n_clusters}")
    print(f"  Points per cluster: {args.points_per_cluster}")
    
    # Initialize models and training
    gmvae_model = None
    vae_model = None
    gmvae_losses = None
    vae_losses = None
    gmvae_metrics = None
    vae_metrics = None
    
    # Train GMVAE
    if args.model_type in ['gmvae', 'both']:
        if args.latent_dim == 1:
            # Use true cluster centers for 1D
            fixed_means = true_centers
        else:
            # Generate fixed means for 2D (circular arrangement)
            fixed_means = []
            for k in range(args.n_clusters):
                angle = 2 * math.pi * k / args.n_clusters
                fixed_means.append([args.radius * math.cos(angle), args.radius * math.sin(angle)])
            fixed_means = torch.tensor(fixed_means, dtype=torch.float32)
        
        gmvae_model = GMVAE(
            n_input=args.n_features,
            n_latent=args.latent_dim,
            fixed_means=fixed_means,
            prior_sigma=args.prior_sigma,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            likelihood="nb"
        ).to(device)
        
        gmvae_trainer = Trainer(gmvae_model, device, args, "GMVAE")
        gmvae_results = gmvae_trainer.train(X)
        gmvae_model = gmvae_results['model']
        gmvae_losses = gmvae_results['loss_history']
        gmvae_metrics = gmvae_trainer.evaluate(X, labels, Z_true)
    
    # Train VAE
    if args.model_type in ['vae', 'both']:
        vae_model = VAE(
            n_input=args.n_features,
            n_latent=args.latent_dim,
            n_hidden=args.n_hidden,
            n_layers=args.n_layers,
            likelihood="nb"
        ).to(device)
        
        vae_trainer = Trainer(vae_model, device, args, "VAE")
        vae_results = vae_trainer.train(X)
        vae_model = vae_results['model']
        vae_losses = vae_results['loss_history']
        vae_metrics = vae_trainer.evaluate(X, labels, Z_true)
    
    # Plotting and analysis
    print(f"\n=== Generating Visualizations ===")
    
    # Training losses
    plot_training_losses(gmvae_losses, vae_losses, 
                        save_path=args.save_plots + "_training_losses.png" if args.save_plots else None)
    
    # Latent space comparison
    plot_latent_space_comparison(Z_true, gmvae_metrics, vae_metrics, labels, 
                                args.latent_dim,
                                save_path=args.save_plots + "_latent_comparison.png" if args.save_plots else None)
    
    # Encoded latent distribution plots using Trainer methods
    if gmvae_metrics:
        gmvae_trainer.plot_encoded_latent_distribution(gmvae_metrics, labels,
                                       save_path=args.save_plots + "_gmvae_latent_distribution.png" if args.save_plots else None)
    
    if vae_metrics:
        vae_trainer.plot_encoded_latent_distribution(vae_metrics, labels,
                                       save_path=args.save_plots + "_vae_latent_distribution.png" if args.save_plots else None)
    
    # Clustering analysis (GMVAE only)
    if gmvae_metrics:
        gmvae_trainer.plot_clustering_analysis(gmvae_metrics, labels,
                               save_path=args.save_plots + "_clustering_analysis.png" if args.save_plots else None)
    
    # Print summary
    print_comparison_summary(gmvae_metrics, vae_metrics, args)
    
    return gmvae_model, vae_model, gmvae_metrics, vae_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation and testing for VAE and GMVAE models")
    
    # Data generation arguments
    parser.add_argument("--n_clusters", '-K', type=int, default=4,
                       help="Number of clusters")
    parser.add_argument("--points_per_cluster", type=int, default=300,
                       help="Points per cluster")
    parser.add_argument("--n_features", type=int, default=100,
                       help="Number of features")
    parser.add_argument("--latent_dim", '-d', type=int, default=1,
                       help="Latent dimension (1 or 2)")
    parser.add_argument("--theta_val", type=float, default=12.0,
                       help="Negative binomial dispersion parameter")
    parser.add_argument("--cluster_centers", type=float, nargs='+', default=None,
                       help="Cluster centers for 1D (if None, auto-generated)")
    parser.add_argument("--radius", type=float, default=3.0,
                       help="Radius for circular cluster arrangement (2D)")
    parser.add_argument("--cluster_spread", type=float, default=0.2,
                       help="Standard deviation of cluster points")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="gmvae", 
                       choices=['vae', 'gmvae', 'both'],
                       help="Model type to train")
    parser.add_argument("--n_hidden", '-H', type=int, default=128,
                       help="Number of hidden units")
    parser.add_argument("--n_layers", '-L', type=int, default=2,
                       help="Number of hidden layers")
    parser.add_argument("--prior_sigma", type=float, default=0.2,
                       help="Prior sigma for GMVAE")
    
    # Training arguments
    parser.add_argument("--n_epochs", '-e', type=int, default=300,
                       help="Number of training epochs")
    parser.add_argument("--kl_warmup_epochs", type=int, default=50,
                       help="Number of warmup epochs for KL annealing")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--print_every", type=int, default=20,
                       help="Print training progress every N epochs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_plots", type=str, default=None,
                       help="Prefix for saving plots (if None, plots are not saved)")
    
    # GMVAE stabilization arguments
    parser.add_argument("--entropy_warmup_epochs", type=int, default=20,
                       help="Epochs to keep a small entropy bonus on q(c|x)")
    parser.add_argument("--lambda_entropy", type=float, default=1e-3,
                       help="Weight for early entropy bonus on q(c|x)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.latent_dim not in [1, 2]:
        raise ValueError("latent_dim must be 1 or 2")
    
    if args.model_type == 'both' and args.latent_dim == 2:
        print("Warning: Comparing VAE and GMVAE in 2D may not be meaningful for clustering evaluation")
    
    print("Parsed arguments:", args)
    
    # Run simulation
    gmvae_model, vae_model, gmvae_metrics, vae_metrics = run_simulation(args)
