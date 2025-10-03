#!/usr/bin/env python3
"""
Training and Evaluation Framework for VAE and GMVAE Models

This module provides a unified training and evaluation framework that can be used
across different scripts and experiments. It abstracts common training procedures,
evaluation metrics, and visualization functions.

Usage:
    from trainer import Trainer
    
    trainer = Trainer(model, device, args)
    trainer.train(X, batch_index=batch_index)
    metrics = trainer.evaluate(X, labels, Z_true)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
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

class Trainer:
    """
    Unified training and evaluation framework for VAE and GMVAE models.
    
    This class provides:
    - Standardized training loops with KL annealing
    - Comprehensive evaluation metrics
    - Visualization functions
    - Early stopping and model checkpointing
    - Component usage analysis (for GMVAE)
    """
    
    def __init__(self, model, device, args, model_name="Model"):
        """
        Initialize the trainer.
        
        Args:
            model: The VAE/GMVAE model to train
            device: Device to use for training
            args: Training arguments (should have attributes like n_epochs, learning_rate, etc.)
            model_name: Name for logging purposes
        """
        self.model = model
        self.device = device
        self.args = args
        self.model_name = model_name
        
        # Training state
        self.optimizer = None
        self.loss_history = []
        self.kl_weight_history = []
        self.best_loss = float('inf')
        self.bad_epochs = 0
        
        # Initialize optimizer
        self._setup_optimizer()
        
    def _setup_optimizer(self):
        """Setup the optimizer."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=getattr(self.args, 'learning_rate', 5e-4),
            weight_decay=getattr(self.args, 'weight_decay', 1e-5)
        )
    
    def _compute_kl_weight(self, epoch: int) -> float:
        """Compute KL weight with annealing."""
        n_warmup_epochs = getattr(self.args, 'kl_warmup_epochs', 50)
        prior_weight = getattr(self.args, 'prior_weight', 1.0)
        
        if epoch < n_warmup_epochs:
            return prior_weight * epoch / n_warmup_epochs
        else:
            return prior_weight
    
    def _fit_step(self, xb, batch_index=None, kl_weight=1.0, epoch=0):
        """Single training step with GMVAE stabilization."""
        self.optimizer.zero_grad()
        
        # Forward pass
        if batch_index is not None:
            forward_output = self.model(xb, batch_index=batch_index)
        else:
            forward_output = self.model(xb)
        
        # GMVAE-specific stabilized training
        if hasattr(self.model, 'n_components'):  # GMVAE
            logits_c, mu_q, logvar_q = forward_output["latent_params"]
            
            # --- (A) Use β_c=1.0; β_z warms up ---
            t = epoch / getattr(self.args, 'kl_warmup_epochs', 50)
            beta_c = 1.0
            beta_z = min(1.0, t)
            
            # Reconstruction loss
            if hasattr(self.model, "_expected_recon_over_c"):
                recon = self.model._expected_recon_over_c(xb, logits_c, mu_q, logvar_q)  # (B,)
            else:
                # Fall back to BaseVAE recon via model.loss, then extract only recon
                tmp = self.model.loss(xb, forward_output, kl_weight=0.0)
                recon = tmp["recon_loss"]  # (scalar averaged)
                if not torch.is_tensor(recon):
                    recon = torch.tensor(recon, device=xb.device)
            
            # KL split
            kl_c, kl_z = self.model._kl_divergence_split(logits_c, mu_q, logvar_q)  # (B,)
            
            # Base loss (average over batch)
            loss = (recon + beta_c * kl_c + beta_z * kl_z).mean()
            
            # --- (B) Optional: entropy bonus for first N epochs ---
            entropy_warmup_epochs = getattr(self.args, 'entropy_warmup_epochs', 0)
            lambda_entropy = getattr(self.args, 'lambda_entropy', 1e-3)
            
            if entropy_warmup_epochs > 0 and epoch <= entropy_warmup_epochs:
                probs = torch.softmax(logits_c, dim=-1)
                ent = (-(probs.clamp_min(1e-9).log() * probs).sum(dim=-1)).mean()
                loss = loss - lambda_entropy * ent  # maximize entropy (tiny λ)
            
            # Pack a dict for logging
            loss_dict = {
                "loss": loss,
                "recon_loss": recon.mean() if recon.ndim > 0 else recon,
                "kl_c": kl_c.mean(),
                "kl_z": kl_z.mean(),
                "kl_local": (kl_c + kl_z).mean(),  # for compatibility with existing plots
                "beta_c": torch.tensor(beta_c),
                "beta_z": torch.tensor(beta_z),
            }
        else:
            # VAE path unchanged
            loss_dict = self.model.loss(xb, forward_output, kl_weight=kl_weight)
        
        # Backward pass
        loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        
        return loss_dict
    
    def train(self, X, batch_index=None, early_stopping=True, patience=80):
        """
        Train the model.
        
        Args:
            X: Input data tensor
            batch_index: Batch indices for batch correction (optional)
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait before early stopping
        
        Returns:
            dict: Training history and final model
        """
        print(f"\n=== Training {self.model_name} ===")
        
        # Setup data loader
        if batch_index is not None:
            dataset = TensorDataset(X.float(), batch_index)
            dl = DataLoader(dataset, batch_size=getattr(self.args, 'batch_size', 128), shuffle=True)
        else:
            dataset = TensorDataset(X.float())
            dl = DataLoader(dataset, batch_size=getattr(self.args, 'batch_size', 128), shuffle=True)
        
        n_epochs = getattr(self.args, 'n_epochs', 300)
        print_every = getattr(self.args, 'print_every', 20)
        
        print(f"Starting training for {n_epochs} epochs...")
        
        # Component usage analysis for GMVAE (if applicable)
        if hasattr(self.model, 'n_components'):
            self._analyze_component_usage(X, batch_index, epoch=0)
        
        # Training loop
        for epoch in range(n_epochs):
            self.model.train()
            
            # KL weight annealing
            kl_weight = self._compute_kl_weight(epoch)
            self.kl_weight_history.append(kl_weight)
            
            # Training step
            epoch_losses = {}
            total_loss = 0.0
            n_batches = 0
            
            for batch in dl:
                if batch_index is not None:
                    xb, batch_idx = batch
                    xb = xb.to(self.device)
                    batch_idx = batch_idx.to(self.device)
                    loss_dict = self._fit_step(xb, batch_idx, kl_weight, epoch)
                else:
                    (xb,) = batch
                    xb = xb.to(self.device)
                    loss_dict = self._fit_step(xb, None, kl_weight, epoch)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item() * xb.size(0)
                
                total_loss += loss_dict["loss"].item() * xb.size(0)
                n_batches += xb.size(0)
            
            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
            total_loss /= n_batches
            
            self.loss_history.append(epoch_losses)
            
            # Early stopping and checkpointing
            if total_loss < self.best_loss - 1e-4:
                self.best_loss = total_loss
                self.bad_epochs = 0
                torch.save(self.model.state_dict(), "best.pt")
            else:
                self.bad_epochs += 1
            
            if early_stopping and self.bad_epochs > patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if epoch == 0 or (epoch + 1) % print_every == 0 or epoch == n_epochs - 1:
                print(f"[{epoch+1:03d}] loss={total_loss:.3f}  kl_w={kl_weight:.2f}  "
                      f"recon={epoch_losses.get('recon_loss', 0):.3f}  "
                      f"kl={epoch_losses.get('kl_local', 0):.3f}")
                
                # Show beta values for GMVAE
                if hasattr(self.model, 'n_components'):
                    beta_c = epoch_losses.get('beta_c', 0)
                    beta_z = epoch_losses.get('beta_z', 0)
                    print(f"    β_c={beta_c:.2f}, β_z={beta_z:.2f}")
                
                # Periodic component usage analysis
                if hasattr(self.model, 'n_components') and (epoch + 1) % print_every == 0:
                    self._analyze_component_usage(X, batch_index, epoch=epoch+1)
        
        print(f"{self.model_name} training completed!")
        
        return {
            'model': self.model,
            'loss_history': self.loss_history,
            'kl_weight_history': self.kl_weight_history,
            'best_loss': self.best_loss
        }
    
    def _analyze_component_usage(self, X, batch_index=None, epoch=None):
        """Analyze component usage for GMVAE models with enhanced logging."""
        if epoch is not None:
            print(f"\n=== Component Usage Analysis (Epoch {epoch}) ===")
        else:
            print(f"\n=== Component Usage Analysis ===")
        
        self.model.eval()
        with torch.no_grad():
            if batch_index is not None:
                logits_c, mu_q, logvar_q = self.model._get_latent_params(X.float().to(self.device), batch_index)
            else:
                logits_c, mu_q, logvar_q = self.model._get_latent_params(X.float().to(self.device))
            
            probs = torch.softmax(logits_c, -1)
            usage = probs.mean(0).cpu().numpy()
            H = (-probs.clamp_min(1e-9).log() * probs).sum(-1).mean().item()
            logK = np.log(self.model.n_components)
            
            print(f"Usage: {np.round(usage, 3)}")
            print(f"Mean entropy: {H:.3f}")
            print(f"Log K: {logK:.3f}")
            
            # Check if usage is close to uniform
            uniform_usage = 1.0 / self.model.n_components
            max_deviation = np.max(np.abs(usage - uniform_usage))
            print(f"Max deviation from uniform: {max_deviation:.3f} (uniform = {uniform_usage:.3f})")
    
    def evaluate(self, X, labels, Z_true=None, batch_index=None):
        """
        Evaluate the trained model.
        
        Args:
            X: Input data tensor
            labels: Ground truth labels
            Z_true: True latent values (optional, for latent space recovery analysis)
            batch_index: Batch indices for batch correction (optional)
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n=== Evaluating {self.model_name} ===")
        
        self.model.eval()
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            if isinstance(self.model, type(self.model)) and hasattr(self.model, 'n_components'):
                # GMVAE evaluation
                if batch_index is not None:
                    latent_params = self.model._get_latent_params(X.float().to(device), batch_index)
                else:
                    latent_params = self.model._get_latent_params(X.float().to(device))
                
                logits_c, mu_q, logvar_q = latent_params
                
                # Get cluster assignments
                cluster_probs = torch.softmax(logits_c, dim=-1)
                predicted_clusters = torch.argmax(cluster_probs, dim=-1)
                
                # Get latent representations (mean of most likely component)
                batch_size = logits_c.shape[0]
                idx = predicted_clusters.view(batch_size, 1, 1).expand(batch_size, 1, self.model.n_latent)
                enc_mu = torch.gather(mu_q, 1, idx).squeeze(1)
                
                # Move to CPU
                enc_mu = enc_mu.cpu()
                predicted_clusters = predicted_clusters.cpu()
                cluster_probs = cluster_probs.cpu()
                
                # Clustering metrics
                ari = adjusted_rand_score(labels.numpy(), predicted_clusters.numpy())
                nmi = normalized_mutual_info_score(labels.numpy(), predicted_clusters.numpy())
                cm = confusion_matrix(labels.numpy(), predicted_clusters.numpy())
                
                metrics = {
                    'enc_mu': enc_mu,
                    'predicted_clusters': predicted_clusters,
                    'cluster_probs': cluster_probs,
                    'ari': ari,
                    'nmi': nmi,
                    'confusion_matrix': cm,
                    'mean_confidence': torch.mean(torch.max(cluster_probs, dim=1)[0]).item()
                }
                
            else:
                # VAE evaluation
                if batch_index is not None:
                    mu, logvar = self.model._get_latent_params(X.float().to(device), batch_index)
                else:
                    mu, logvar = self.model._get_latent_params(X.float().to(device))
                
                enc_mu = mu.cpu()
                
                metrics = {
                    'enc_mu': enc_mu,
                    'predicted_clusters': None,
                    'cluster_probs': None,
                    'ari': None,
                    'nmi': None,
                    'confusion_matrix': None,
                    'mean_confidence': None
                }
        
        # Latent space recovery metrics (if Z_true provided)
        if Z_true is not None:
            correlation, p_value = pearsonr(Z_true.numpy().flatten(), enc_mu.numpy().flatten())
            residuals = enc_mu.numpy().flatten() - Z_true.numpy().flatten()
            
            metrics.update({
                'correlation': correlation,
                'p_value': p_value,
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals)
            })
        
        # Print results
        print(f"{self.model_name} evaluation completed:")
        if 'correlation' in metrics:
            print(f"  Latent space correlation: {metrics['correlation']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
        if metrics['ari'] is not None:
            print(f"  ARI: {metrics['ari']:.4f}")
            print(f"  NMI: {metrics['nmi']:.4f}")
            print(f"  Mean cluster confidence: {metrics['mean_confidence']:.4f}")
        
        return metrics
    
    def plot_training_losses(self, save_path=None):
        """Plot training loss curves."""
        if not self.loss_history:
            print("No training history available for plotting.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        ax = axes[0]
        total_losses = [h['loss'] for h in self.loss_history]
        ax.plot(total_losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'{self.model_name} - Training Loss')
        ax.grid(True, alpha=0.3)
        
        # Reconstruction loss
        ax = axes[1]
        recon_losses = [h.get('recon_loss', 0) for h in self.loss_history]
        ax.plot(recon_losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Reconstruction Loss')
        ax.set_title(f'{self.model_name} - Reconstruction Loss')
        ax.grid(True, alpha=0.3)
        
        # KL loss
        ax = axes[2]
        kl_losses = [h.get('kl_local', 0) for h in self.loss_history]
        ax.plot(kl_losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('KL Loss')
        ax.set_title(f'{self.model_name} - KL Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_encoded_latent_distribution(self, metrics, labels, save_path=None):
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
        ax.set_title(f'Encoded Latent Distribution - {self.model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clustering_analysis(self, metrics, labels, save_path=None):
        """Plot clustering-specific analysis for GMVAE."""
        if metrics['ari'] is None:
            print("Clustering analysis only available for GMVAE models.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confusion matrix
        ax = axes[0, 0]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted Cluster')
        ax.set_ylabel('True Cluster')
        ax.set_title('Confusion Matrix')
        
        # Cluster assignment confidence
        ax = axes[0, 1]
        max_probs = torch.max(metrics['cluster_probs'], dim=1)[0]
        ax.hist(max_probs.numpy(), bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Max Cluster Probability')
        ax.set_ylabel('Frequency')
        ax.set_title('Cluster Assignment Confidence')
        ax.grid(True, alpha=0.3)
        
        # Predicted vs True clusters
        ax = axes[1, 0]
        predicted = metrics['predicted_clusters'].numpy()
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
        metric_names = ['ARI', 'NMI', 'Confidence']
        values = [metrics['ari'], metrics['nmi'], metrics['mean_confidence']]
        bars = ax.bar(metric_names, values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'])
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
    
    def compute_lineage_statistics(self, X, labels, batch_index=None):
        """
        Compute lineage-specific statistics from the trained model.
        
        Args:
            X: Input data tensor
            labels: Ground truth lineage labels
            batch_index: Batch indices for batch correction (optional)
        
        Returns:
            dict: Lineage statistics including means and variances
        """
        print(f"\n=== Computing Lineage Statistics ===")
        
        self.model.eval()
        with torch.no_grad():
            # Get latent representations from trained model
            if batch_index is not None:
                forward_output = self.model(X, batch_index=batch_index)
            else:
                forward_output = self.model(X)
            
            z = forward_output['z']  # (N, latent_dim)
            
            # Get latent parameters for each component (if GMVAE)
            if hasattr(self.model, 'n_components'):
                logits_c, mu_q, logvar_q = forward_output['latent_params']
                
                # Get component assignments (most likely component for each cell)
                probs_c = torch.softmax(logits_c, dim=-1)
                assigned_components = torch.argmax(probs_c, dim=-1)  # (N,)
            else:
                assigned_components = None
            
            # Group by lineage and compute statistics
            lineage_stats = {}
            unique_lineages = np.unique(labels)
            
            for lineage in unique_lineages:
                lineage_mask = labels == lineage
                lineage_z = z[lineage_mask]
                
                if len(lineage_z) > 0:
                    # Compute mean and variance for this lineage
                    lineage_mean = torch.mean(lineage_z, dim=0)
                    lineage_var = torch.var(lineage_z, dim=0)
                    
                    stats = {
                        'mean': lineage_mean,
                        'var': lineage_var,
                        'std': torch.sqrt(lineage_var),
                        'n_cells': len(lineage_z),
                        'latent_samples': lineage_z
                    }
                    
                    # Add component distribution for GMVAE
                    if assigned_components is not None:
                        lineage_components = assigned_components[lineage_mask]
                        component_counts = torch.bincount(lineage_components, minlength=self.model.n_components)
                        component_probs = component_counts.float() / len(lineage_components)
                        stats['component_probs'] = component_probs
                    
                    lineage_stats[lineage] = stats
                    
                    print(f"Lineage {lineage}: {len(lineage_z)} cells, mean={lineage_mean.numpy()}, std={torch.sqrt(lineage_var).numpy()}")
        
        return lineage_stats
