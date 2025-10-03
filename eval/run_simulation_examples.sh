#!/bin/bash

# Example usage of the simulation script
# This script demonstrates various ways to run the simulation and testing

echo "=== VAE/GMVAE Simulation Examples ==="
echo

# Example 1: Basic GMVAE with 1D latent space
echo "Example 1: Basic GMVAE with 1D latent space"
echo "Command: python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --n_epochs 200"
echo
python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --n_epochs 200
echo

# Example 2: VAE with 2D latent space
echo "Example 2: VAE with 2D latent space"
echo "Command: python test_simulation.py --model_type vae --n_clusters 5 --latent_dim 2 --n_epochs 150"
echo
python test_simulation.py --model_type vae --n_clusters 5 --latent_dim 2 --n_epochs 150
echo

# Example 3: Compare both models
echo "Example 3: Compare VAE and GMVAE"
echo "Command: python test_simulation.py --model_type both --n_clusters 4 --latent_dim 1 --n_epochs 100"
echo
python test_simulation.py --model_type both --n_clusters 4 --latent_dim 1 --n_epochs 100
echo

# Example 4: Custom cluster centers
echo "Example 4: Custom cluster centers"
echo "Command: python test_simulation.py --model_type gmvae --n_clusters 3 --latent_dim 1 --cluster_centers -1.5 0 1.5 --n_epochs 150"
echo
python test_simulation.py --model_type gmvae --n_clusters 3 --latent_dim 1 --cluster_centers -1.5 0 1.5 --n_epochs 150
echo

# Example 5: Save plots
echo "Example 5: Save plots to files"
echo "Command: python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --save_plots results/gmvae_4clusters"
echo
mkdir -p results
python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --save_plots results/gmvae_4clusters
echo

echo "=== All examples completed ==="
