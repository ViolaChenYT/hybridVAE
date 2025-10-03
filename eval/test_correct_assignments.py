from re import S
import numpy as np
import pandas as pd
import scanpy as sc
import os, random
import sys
module_path = os.path.abspath(os.path.join('.', 'src'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from gmvae import GMVAE
from circular_vae import CircularVAE
from base_vae import BaseVAE
from vae import VAE
from utils import *
from trainer import Trainer, set_seed
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import seaborn as sns
from collections import defaultdict

# set_seed function is now imported from trainer module
set_seed()

def load_h5ad_data(file_path, label_key):
    """
    Load h5ad file and extract data and labels.
    Automatically uses adata.raw.X if available, otherwise uses adata.X.
    
    Args:
        file_path (str): Path to the h5ad file
        label_key (str): Key in adata.obs containing the labels
    
    Returns:
        tuple: (X_tensor, labels, adata)
    """
    print(f"Loading data from: {file_path}")
    adata = sc.read_h5ad(file_path)
    
    # Extract data - automatically use raw if available
    if adata.raw is not None:
        X = adata.raw.X.toarray() if hasattr(adata.raw.X, 'toarray') else adata.raw.X
        print(f"Using raw data with shape: {X.shape}")
    else:
        X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        print(f"Using processed data with shape: {X.shape}")
    
    # Extract labels
    if label_key not in adata.obs.columns:
        raise ValueError(f"Label key '{label_key}' not found in adata.obs. Available keys: {list(adata.obs.columns)}")
    
    labels = adata.obs[label_key].values
    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Filter out zero-sum rows
    row_sums = X_tensor.sum(axis=1)
    keep_indices = (row_sums > 0)
    X_tensor = X_tensor[keep_indices]
    labels = labels[keep_indices.cpu().numpy()] if isinstance(keep_indices, torch.Tensor) else labels[keep_indices]
    
    print(f"Data shape after filtering: {X_tensor.shape}")
    print(f"Data range: [{X_tensor.min().item():.3f}, {X_tensor.max().item():.3f}]")
    
    return X_tensor, labels, adata

def extract_X_and_batch(adata, layer: str = None, batch_key: str = None):
    # X matrix
    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):  # sparse
        X = X.toarray()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)

    # batch index
    if batch_key is None:
        return X_tensor, None, 0, None

    if batch_key not in adata.obs.columns:
        raise KeyError(f"batch_key='{batch_key}' not found in adata.obs")

    col = adata.obs[batch_key].astype("category")
    cats = col.cat.categories
    codes = col.cat.codes.to_numpy()  # -1 if NaN
    if (codes < 0).any():
        raise ValueError(f"Found NaN/undefined in adata.obs['{batch_key}']. Fix or drop.")
    batch_index = torch.as_tensor(codes, dtype=torch.long)
    return X_tensor, batch_index, len(cats), list(map(str, cats))

def create_fixed_priors(prior_values, n_components, latent_dim=1):
    """
    Create fixed priors from command line arguments.

    Args:
        prior_values (str): Comma-separated string of prior values
        n_components (int): Number of components
        latent_dim (int): Latent dimension (d)

    Returns:
        torch.Tensor: Fixed means tensor of shape (K, d)
    """
    if prior_values is None and n_components is None:
        raise ValueError("Either --prior_values or --n_components must be specified")

    prior_list = None
    if prior_values is not None:
        # prior_values is a list of tokens due to nargs="+"
        tokens = []
        for tok in prior_values:
            tokens.extend(tok.split(","))   # split comma blobs
        try:
            prior_list = [float(x.strip()) for x in tokens if x.strip() != ""]
        except ValueError:
            raise ValueError(f"Invalid prior values. Use numbers like: -p -2 -1 0 1 2  or  -p -2,-1,0,1,2")
        if n_components is not None:
            if len(prior_list) % n_components != 0:
                raise ValueError(f"Number of prior values ({len(prior_list)}) is not a multiple of --n_components ({n_components})")
            d = len(prior_list) // n_components
            if latent_dim is not None and d != latent_dim:
                raise ValueError(f"Latent dimension inferred from prior values ({d}) does not match specified latent_dim ({latent_dim})")
            latent_dim = d
        else:
            if latent_dim == 1:
                n_components = len(prior_list)
            else:
                raise ValueError("If --n_components is not specified, prior_values must be a flat list of length K*d")
        fixed_means = torch.tensor(prior_list, dtype=torch.float32).reshape(n_components, latent_dim)
    else:
        # Default: linearly spaced means along the first axis, zeros elsewhere
        fixed_means = torch.zeros(n_components, latent_dim, dtype=torch.float32)
        fixed_means[:, 0] = torch.linspace(-2.0, 2.0, n_components, dtype=torch.float32)

    print(f"Using fixed priors:\n{fixed_means.numpy()}")
    print(f"Number of components: {fixed_means.shape[0]}, latent dim: {fixed_means.shape[1]}")
    return fixed_means

def train_gmvae(X_tensor, fixed_means, args, n_batches=0, batch_index=None, early_stopping=True):
    """
    Train GMVAE model using the Trainer class.
    
    Args:
        X_tensor (torch.Tensor): Input data
        fixed_means (torch.Tensor): Fixed prior means
        args: Command line arguments
        n_batches: Number of batches for batch correction
        batch_index: Batch indices
        early_stopping: Whether to use early stopping
    
    Returns:
        tuple: (trained_model, loss_history, kl_weight_history)
    """
    print(f"\n=== Training GMVAE ===")
    print(f"Input dimension: {X_tensor.shape[1]}")
    print(f"Latent dimension: {args.n_latent}")
    print(f"Number of components: {len(fixed_means)}")
    
    if torch.cuda.is_available(): 
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): 
        device = torch.device("mps")
    else: 
        device = torch.device("cpu")

    # Initialize model
    model = GMVAE(
        n_input=X_tensor.shape[1],
        n_latent=args.n_latent,
        fixed_means=fixed_means,
        n_batches=n_batches,
        batch_emb_dim=8,
        batch_index=batch_index
    ).to(device)
    
    # Use Trainer class
    trainer = Trainer(model, device, args, "GMVAE")
    results = trainer.train(X_tensor, batch_index=batch_index, early_stopping=early_stopping)
    
    return results['model'], results['loss_history'], results['kl_weight_history']

def compute_lineage_statistics(model, X_tensor, labels, batch_index=None):
    """
    Compute lineage-specific statistics from the trained model using Trainer class.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        labels: Ground truth lineage labels
        batch_index: Batch indices for batch correction
    
    Returns:
        dict: Lineage statistics including means and variances
    """
    device = next(model.parameters()).device
    trainer = Trainer(model, device, None, "GMVAE")
    return trainer.compute_lineage_statistics(X_tensor, labels, batch_index)

def generate_correct_assignments(model, X_tensor, labels, lineage_stats, batch_index=None, n_samples=1):
    """
    Generate 'correct' latent assignments by sampling from lineage-specific distributions.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        labels: Ground truth lineage labels
        lineage_stats: Statistics computed from compute_lineage_statistics
        batch_index: Batch indices for batch correction
        n_samples: Number of samples to generate per cell
    
    Returns:
        torch.Tensor: Correct latent assignments (N, latent_dim)
    """
    print(f"\n=== Generating Correct Assignments ===")
    
    device = next(model.parameters()).device
    n_cells = len(labels)
    latent_dim = model.n_latent
    
    # Initialize tensor for correct assignments
    correct_assignments = torch.zeros(n_cells, latent_dim, device=device)
    
    unique_lineages = np.unique(labels)
    
    for lineage in unique_lineages:
        lineage_mask = labels == lineage
        lineage_indices = np.where(lineage_mask)[0]
        
        if lineage in lineage_stats:
            stats = lineage_stats[lineage]
            lineage_mean = stats['mean'].to(device)
            lineage_std = stats['std'].to(device)
            
            # Sample from N(μ_i, σ_i) for each cell in this lineage
            for idx in lineage_indices:
                # Sample from the lineage-specific distribution
                sample = torch.normal(lineage_mean, lineage_std)
                correct_assignments[idx] = sample
    
    print(f"Generated correct assignments for {n_cells} cells")
    return correct_assignments

def compute_correct_assignment_loss(model, X_tensor, correct_assignments, batch_index=None):
    """
    Compute reconstruction loss using correct latent assignments.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        correct_assignments (torch.Tensor): Correct latent assignments
        batch_index: Batch indices for batch correction
    
    Returns:
        dict: Loss components
    """
    print(f"\n=== Computing Correct Assignment Loss ===")
    
    model.eval()
    with torch.no_grad():
        # Use the correct assignments as latent representations
        z_correct = correct_assignments
        
        # Prepare decoder input (with batch correction if needed)
        if batch_index is not None and model.n_batches > 0:
            b = model.batch_emb(batch_index)
            z_in = model.dec_cond(torch.cat([z_correct, b], dim=1))
        else:
            z_in = z_correct
        
        # Forward through decoder
        hidden_decoder = model.decoder(z_in)
        
        # Get reconstruction parameters
        if model.likelihood == "nb":
            px_scale = torch.softmax(model.px_scale_decoder(hidden_decoder), dim=-1)
            px_r = torch.exp(model.px_r_decoder(hidden_decoder))
            
            # Compute library size
            library = torch.sum(X_tensor, dim=1, keepdim=True).clamp_min(1e-8)
            px_rate = library * px_scale
            
            # Compute reconstruction loss (negative log-likelihood)
            from torch.distributions import NegativeBinomial
            p = px_r / (px_r + px_rate + 1e-8)
            nb_dist = NegativeBinomial(total_count=px_r, probs=p)
            recon_loss = -nb_dist.log_prob(X_tensor).sum(dim=-1)
            
        else:  # Gaussian likelihood
            px_mu = model.px_mu_decoder(hidden_decoder)
            if model.gaussian_homoscedastic:
                px_logvar = torch.zeros_like(px_mu)
            else:
                px_logvar = model.px_logvar_decoder(hidden_decoder)
            
            # Compute reconstruction loss
            from torch.distributions import Normal
            px_std = torch.exp(0.5 * px_logvar)
            normal_dist = Normal(px_mu, px_std)
            recon_loss = -normal_dist.log_prob(X_tensor).sum(dim=-1)
        
        # No KL term for correct assignments (we're not using the encoder)
        total_loss = recon_loss.mean()
        recon_loss_mean = recon_loss.mean()
        
        loss_dict = {
            'loss': total_loss,
            'recon_loss': recon_loss_mean,
            'kl_local': torch.tensor(0.0),  # No KL for correct assignments
            'kl_global': torch.tensor(0.0)
        }
        
        print(f"Correct assignment loss: {total_loss.item():.4f}")
        print(f"Reconstruction loss: {recon_loss_mean.item():.4f}")
        
        return loss_dict

def visualize_assignments_comparison(model, X_tensor, labels, lineage_stats, batch_index=None, save_path=None):
    """
    Create a frequency distribution plot comparing correct assignments vs learned optimal assignments.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        labels: Ground truth lineage labels
        lineage_stats: Statistics computed from compute_lineage_statistics
        batch_index: Batch indices for batch correction
        save_path: Optional path to save the figure
    
    Returns:
        tuple: (trained_assignments, correct_assignments) for further analysis
    """
    print(f"\n=== Creating Assignment Comparison Visualization ===")
    
    model.eval()
    with torch.no_grad():
        # Get learned optimal assignments from trained VAE
        trained_output = model(X_tensor, batch_index=batch_index)
        trained_assignments = trained_output['z'].numpy().flatten()
        
        # Generate correct assignments
        correct_assignments_tensor = generate_correct_assignments(model, X_tensor, labels, lineage_stats, batch_index)
        correct_assignments = correct_assignments_tensor.numpy().flatten()
    
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    unique_lineages = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_lineages)))
    
    # Plot 1: Learned Optimal Assignments (Trained VAE)
    for i, lineage in enumerate(unique_lineages):
        mask = labels == lineage
        lineage_trained = trained_assignments[mask]
        
        # Create histogram
        counts, bins = np.histogram(lineage_trained, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax1.plot(bin_centers, counts, color=colors[i], label=lineage, linewidth=2, alpha=0.8)
        ax1.fill_between(bin_centers, counts, alpha=0.3, color=colors[i])
    
    ax1.set_title('Learned Optimal Assignments (Trained VAE)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Latent Coordinate Value', fontsize=12)
    ax1.set_ylabel('Normalized Frequency', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Correct Assignments (Lineage-specific sampling)
    for i, lineage in enumerate(unique_lineages):
        mask = labels == lineage
        lineage_correct = correct_assignments[mask]
        
        # Create histogram
        counts, bins = np.histogram(lineage_correct, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax2.plot(bin_centers, counts, color=colors[i], label=lineage, linewidth=2, alpha=0.8)
        ax2.fill_between(bin_centers, counts, alpha=0.3, color=colors[i])
    
    ax2.set_title('Correct Assignments (Lineage-specific Sampling)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Latent Coordinate Value', fontsize=12)
    ax2.set_ylabel('Normalized Frequency', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Assignment Statistics Summary ===")
    print(f"{'Lineage':<15} {'Trained Mean':<12} {'Trained Std':<12} {'Correct Mean':<12} {'Correct Std':<12}")
    print("-" * 70)
    
    for lineage in unique_lineages:
        mask = labels == lineage
        trained_mean = np.mean(trained_assignments[mask])
        trained_std = np.std(trained_assignments[mask])
        correct_mean = np.mean(correct_assignments[mask])
        correct_std = np.std(correct_assignments[mask])
        
        print(f"{lineage:<15} {trained_mean:<12.3f} {trained_std:<12.3f} {correct_mean:<12.3f} {correct_std:<12.3f}")
    
    return trained_assignments, correct_assignments

def compare_losses(model, X_tensor, labels, batch_index=None, create_visualization=True, save_path=None):
    """
    Compare losses between trained VAE and correct assignments.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        labels: Ground truth lineage labels
        batch_index: Batch indices for batch correction
        create_visualization: Whether to create the comparison plot
        save_path: Optional path to save the visualization
    
    Returns:
        dict: Comparison results
    """
    print(f"\n=== Comparing Losses ===")
    
    # 1. Compute lineage statistics
    lineage_stats = compute_lineage_statistics(model, X_tensor, labels, batch_index)
    
    # 2. Generate correct assignments
    correct_assignments = generate_correct_assignments(model, X_tensor, labels, lineage_stats, batch_index)
    
    # 3. Compute loss with correct assignments
    correct_loss_dict = compute_correct_assignment_loss(model, X_tensor, correct_assignments, batch_index)
    
    # 4. Compute loss with trained VAE
    model.eval()
    with torch.no_grad():
        trained_output = model(X_tensor, batch_index=batch_index)
        trained_loss_dict = model.loss(X_tensor, trained_output, kl_weight=0.0)  # No KL for fair comparison
    
    # 5. Compare results
    comparison = {
        'trained_vae': {
            'total_loss': trained_loss_dict['loss'].item(),
            'recon_loss': trained_loss_dict['recon_loss'].item(),
            'kl_loss': trained_loss_dict['kl_local'].item()
        },
        'correct_assignments': {
            'total_loss': correct_loss_dict['loss'].item(),
            'recon_loss': correct_loss_dict['recon_loss'].item(),
            'kl_loss': correct_loss_dict['kl_local'].item()
        },
        'lineage_stats': lineage_stats
    }
    
    # Compute differences
    comparison['difference'] = {
        'total_loss_diff': comparison['correct_assignments']['total_loss'] - comparison['trained_vae']['total_loss'],
        'recon_loss_diff': comparison['correct_assignments']['recon_loss'] - comparison['trained_vae']['recon_loss']
    }
    
    # Print results
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Trained VAE Loss: {comparison['trained_vae']['total_loss']:.4f}")
    print(f"Correct Assignment Loss: {comparison['correct_assignments']['total_loss']:.4f}")
    print(f"Difference: {comparison['difference']['total_loss_diff']:.4f}")
    print(f"")
    print(f"Trained VAE Recon Loss: {comparison['trained_vae']['recon_loss']:.4f}")
    print(f"Correct Assignment Recon Loss: {comparison['correct_assignments']['recon_loss']:.4f}")
    print(f"Recon Loss Difference: {comparison['difference']['recon_loss_diff']:.4f}")
    
    if comparison['difference']['total_loss_diff'] < 0:
        print(f"\n✓ Correct assignments achieve LOWER loss ({abs(comparison['difference']['total_loss_diff']):.4f} better)")
    else:
        print(f"\n✗ Correct assignments achieve HIGHER loss ({comparison['difference']['total_loss_diff']:.4f} worse)")
    
    # 6. Create visualization if requested
    if create_visualization:
        trained_assignments, correct_assignments_vis = visualize_assignments_comparison(
            model, X_tensor, labels, lineage_stats, batch_index, save_path
        )
        comparison['trained_assignments'] = trained_assignments
        comparison['correct_assignments_vis'] = correct_assignments_vis
    
    return comparison

def run_correct_assignment_test(args):
    """
    Main function to run the correct assignment comparison test.
    """
    # Load data
    X_tensor, labels, adata = load_h5ad_data(args.h5ad_path, args.label_key)
    
    # Create fixed priors
    fixed_means = create_fixed_priors(args.prior_values, args.n_components)
    X, bidx, n_batches, cats = extract_X_and_batch(adata, batch_key=args.batch_key if args.batch_correction else None)
    if args.batch_correction:
        print(f"[batch] Using obs['{args.batch_key}'] with {n_batches} levels: {cats}")

    # Train model
    model, loss_history, kl_weight_history = train_gmvae(X, fixed_means, args, n_batches, bidx)
    
    # Load best model
    if os.path.exists("best.pt"):
        model.load_state_dict(torch.load("best.pt"))
        print("Loaded best model checkpoint")
    
    # Run comparison with visualization
    save_path = f"assignment_comparison_{args.n_components}components.png" if args.save_plot else None
    comparison_results = compare_losses(model, X, labels, bidx, 
                                      create_visualization=args.create_visualization, 
                                      save_path=save_path)
    
    return model, comparison_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained VAE loss with correct latent assignments")
    
    # Data arguments
    parser.add_argument("--h5ad_path", '-ad', type=str, required=True, 
                       help="Path to the h5ad file")
    parser.add_argument("--label_key", '-lk', type=str, default="lineage",
                       help="Key in adata.obs containing the labels")
    parser.add_argument("--batch_correction", action="store_true",
                       help="Whether to perform batch correction")
    parser.add_argument("--batch_key", '-bk', type=str, default="batch",
                       help="Key in adata.obs containing the batch labels")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="gmvae")
    parser.add_argument("--n_latent", '-d', type=int, default=1,
                       help="Number of latent dimensions")
    parser.add_argument("--n_hidden", '-H', type=int, default=128,
                       help="Number of hidden units")
    parser.add_argument("--n_layers", '-L', type=int, default=2,
                       help="Number of hidden layers")
    
    # Training arguments
    parser.add_argument("--n_epochs", '-e', type=int, default=800,
                       help="Number of training epochs")
    parser.add_argument("--n_warmup_epochs", '-w', type=int, default=100,
                       help="Number of warmup epochs for KL annealing")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Learning rate")
    parser.add_argument("--prior_weight", type=float, default=5.0,
                       help="Weight for prior loss")
    parser.add_argument("--print_every", type=int, default=20,
                       help="Print training progress every N epochs")
    
    # Prior arguments
    parser.add_argument("--prior_values", '-p', type=str, default=None,
                       help="Comma-separated list of prior values (e.g., '-2,-1,0,1')", nargs="+")
    parser.add_argument("--n_components", '-K', type=int, default=None,
                       help="Number of components (if not specified, inferred from prior_values)")
    parser.add_argument("--prior_sigma", type=float, default=0.05,
                       help="Prior sigma (not used in current implementation)")
    
    # Visualization arguments
    parser.add_argument("--create_visualization", action="store_true", default=True,
                       help="Create frequency distribution comparison plot")
    parser.add_argument("--save_plot", action="store_true",
                       help="Save the comparison plot to file")
    
    # GMVAE stabilization arguments
    parser.add_argument("--entropy_warmup_epochs", type=int, default=20,
                       help="Epochs to keep a small entropy bonus on q(c|x)")
    parser.add_argument("--lambda_entropy", type=float, default=1e-3,
                       help="Weight for early entropy bonus on q(c|x)")
    
    args = parser.parse_args()
    print("parsed args: ", args)
    
    # Run the analysis
    model, comparison_results = run_correct_assignment_test(args)
