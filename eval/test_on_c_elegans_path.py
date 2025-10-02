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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import seaborn as sns

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN / backends
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Strong determinism (may error if an op has no deterministic impl)
    torch.use_deterministic_algorithms(True, warn_only=True)
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
    
def batch_iter(X, batch_index, batch_size: int, shuffle: bool = True):
    N = X.shape[0]
    order = np.random.permutation(N) if shuffle else np.arange(N)
    for i in range(0, N, batch_size):
        idx = order[i:i+batch_size]
        xb = X[idx]
        if batch_index is None:
            yield xb, None
        else:
            yield xb, batch_index[idx]

def train_gmvae(X_tensor, fixed_means, args, n_batches=0,batch_index=None, early_stopping=True):
    """
    Train GMVAE model.
    
    Args:
        X_tensor (torch.Tensor): Input data
        fixed_means (torch.Tensor): Fixed prior means
        args: Command line arguments
    
    Returns:
        tuple: (trained_model, loss_history, kl_weight_history)
    """
    print(f"\n=== Training GMVAE ===")
    print(f"Input dimension: {X_tensor.shape[1]}")
    print(f"Latent dimension: {args.n_latent}")
    print(f"Number of components: {len(fixed_means)}")
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    # Initialize model
    model = GMVAE(
        n_input=X_tensor.shape[1],
        n_latent=args.n_latent,
        fixed_means=fixed_means,
        n_batches=n_batches,
        batch_emb_dim=8,
        batch_index=batch_index
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training parameters
    n_epochs = args.n_epochs
    n_warmup_epochs = args.n_warmup_epochs
    kl_weight = 0.0
    
    # Lists for tracking
    loss_history = []
    kl_weight_history = []
    
    print(f"Starting training for {n_epochs} epochs...")
    best = float("inf");bad=0;patience = 80
    for epoch in range(n_epochs):
        model.train()

        # KL weight annealing
        if epoch < n_warmup_epochs:
            kl_weight = args.prior_weight * epoch / n_warmup_epochs
        else:
            kl_weight = args.prior_weight

        kl_weight_history.append(kl_weight)
        optimizer.zero_grad()
        out = model(X_tensor, batch_index=batch_index)                  # <- pass experimental batch vector here
        loss_dict = model.loss(X_tensor, out, kl_weight=kl_weight)
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if loss < best - 1e-4:
            best = loss; bad = 0; torch.save(model.state_dict(), "best.pt")
        else: bad += 1
        if early_stopping and bad > patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if (epoch + 1) % args.print_every == 0:
            print(
                f"Epoch [{epoch+1}/{n_epochs}], "
                f"Loss: {loss.item():.4f}, "
                f"KL Weight: {kl_weight:.2f}, "
                f"Recon Loss: {loss_dict['recon_loss']:.4f}, "
                f"KL Local: {loss_dict['kl_local']:.4f}"
            )
    
    print("Training completed!")
    return model, loss_history, kl_weight_history

def plot_1d_latent_space(model, X_tensor, labels, args, batch_index=None):
    """
    Plot 1D latent space visualization.
    
    Args:
        model: Trained GMVAE model
        X_tensor (torch.Tensor): Input data
        labels: Ground truth labels
        args: Command line arguments
        batch_index: Batch indices for batch correction
    """
    print(f"\n=== Generating 1D Latent Space Visualization ===")
    
    # Extract latent representations
    model.eval()
    with torch.no_grad():
        forward_output = model(X_tensor, batch_index=batch_index)
        z = forward_output['z']
    
    # Convert to numpy for plotting
    z_np = z.numpy().flatten()
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'latent_coord': z_np,
        'label': labels
    })
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Box plot by label
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='label', y='latent_coord', ax=ax1)
    ax1.set_title('Distribution of 1D Latent Coordinates by Label')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('1D Latent Coordinate')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Scatter plot colored by label
    ax2 = axes[0, 1]
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax2.scatter(z_np[mask], np.zeros_like(z_np[mask]), 
                   c=[colors[i]], label=str(label), alpha=0.7, s=20)
    ax2.set_title('1D Latent Space (colored by label)')
    ax2.set_xlabel('Latent Coordinate')
    ax2.set_ylabel('')
    ax2.set_yticks([])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Histogram of latent coordinates
    ax3 = axes[1, 0]
    ax3.hist(z_np, bins=50, alpha=0.7, edgecolor='black')
    ax3.set_title('Distribution of Latent Coordinates')
    ax3.set_xlabel('Latent Coordinate')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Violin plot by label
    ax4 = axes[1, 1]
    sns.violinplot(data=df, x='label', y='latent_coord', ax=ax4)
    ax4.set_title('Latent Coordinate Distribution by Label (Violin Plot)')
    ax4.set_xlabel('Label')
    ax4.set_ylabel('1D Latent Coordinate')
    ax4.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("Statistics for each label:")
    for label in sorted(unique_labels):
        label_data = df[df['label'] == label]['latent_coord']
        print(f"Label {label}: mean={label_data.mean():.3f}, std={label_data.std():.3f}, "
              f"min={label_data.min():.3f}, max={label_data.max():.3f}, count={len(label_data)}")

def run_gmvae(args):
    """
    Main function to run GMVAE training and visualization.
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
    
    # Plot results
    plot_1d_latent_space(model, X, labels, args, bidx)
    
    return model, loss_history, kl_weight_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GMVAE on h5ad data and visualize 1D latent space")
    
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
    
    args = parser.parse_args()
    print("parsed args: ", args)
    
    # Run the analysis
    model, loss_history, kl_weight_history = run_gmvae(args)

