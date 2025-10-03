# Simulation and Testing Script

This directory contains a comprehensive simulation and testing script for VAE and GMVAE models, designed to replace the Jupyter notebook workflow with a more robust Python script approach.

## Files

- `test_simulation.py` - Main simulation and testing script
- `run_simulation_examples.sh` - Example usage script
- `README_simulation.md` - This documentation file

## Features

The simulation script provides:

1. **Synthetic Data Generation**
   - 1D and 2D latent space generation
   - Configurable cluster arrangements
   - Negative Binomial count data simulation
   - Ground truth latent values for evaluation

2. **Model Training**
   - VAE (Variational Autoencoder)
   - GMVAE (Gaussian Mixture VAE)
   - Configurable architecture parameters
   - KL annealing and training monitoring

3. **Comprehensive Evaluation**
   - Latent space recovery metrics (correlation, RMSE)
   - Clustering performance (ARI, NMI, confusion matrix)
   - Training loss tracking
   - Statistical significance testing

4. **Rich Visualizations**
   - Training loss curves
   - Latent space comparisons
   - Encoded latent distribution histograms by true cluster
   - Clustering analysis plots
   - Correlation and residual analysis

## Usage

### Basic Usage

```bash
# Train GMVAE with 1D latent space
python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1

# Train VAE with 2D latent space
python test_simulation.py --model_type vae --n_clusters 5 --latent_dim 2

# Compare both models
python test_simulation.py --model_type both --n_clusters 4 --latent_dim 1
```

### Advanced Usage

```bash
# Custom cluster centers
python test_simulation.py --model_type gmvae --n_clusters 3 --latent_dim 1 --cluster_centers -1.5 0 1.5

# Save plots to files
python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --save_plots results/gmvae_4clusters

# Custom training parameters
python test_simulation.py --model_type gmvae --n_clusters 4 --latent_dim 1 --n_epochs 1000 --learning_rate 1e-3
```

### Run Examples

```bash
# Run all example configurations
./run_simulation_examples.sh
```

## Command Line Arguments

### Data Generation
- `--n_clusters` / `-K`: Number of clusters (default: 4)
- `--points_per_cluster`: Points per cluster (default: 300)
- `--n_features`: Number of features (default: 100)
- `--latent_dim` / `-d`: Latent dimension, 1 or 2 (default: 1)
- `--theta_val`: Negative binomial dispersion parameter (default: 12.0)
- `--cluster_centers`: Custom cluster centers for 1D (default: auto-generated)
- `--radius`: Radius for circular cluster arrangement in 2D (default: 3.0)
- `--cluster_spread`: Standard deviation of cluster points (default: 0.2)

### Model Configuration
- `--model_type`: Model type - 'vae', 'gmvae', or 'both' (default: 'gmvae')
- `--n_hidden` / `-H`: Number of hidden units (default: 128)
- `--n_layers` / `-L`: Number of hidden layers (default: 2)
- `--prior_sigma`: Prior sigma for GMVAE (default: 0.25)

### Training Parameters
- `--n_epochs` / `-e`: Number of training epochs (default: 500)
- `--kl_warmup_epochs`: Number of warmup epochs for KL annealing (default: 50)
- `--learning_rate`: Learning rate (default: 5e-4)
- `--batch_size`: Batch size (default: 128)
- `--print_every`: Print training progress every N epochs (default: 20)

### Other Options
- `--seed`: Random seed (default: 42)
- `--save_plots`: Prefix for saving plots (default: None, plots not saved)

## Output

The script provides:

1. **Console Output**
   - Training progress with loss values
   - Model evaluation metrics
   - Comprehensive performance summary

2. **Visualizations**
   - Training loss curves (total, reconstruction, KL)
   - Latent space comparisons (true vs encoded)
   - Correlation plots and residual analysis
   - Clustering performance analysis (GMVAE only)

3. **Saved Files** (if `--save_plots` specified)
   - `{prefix}_training_losses.png`
   - `{prefix}_latent_comparison.png`
   - `{prefix}_gmvae_latent_distribution.png` (if GMVAE trained)
   - `{prefix}_vae_latent_distribution.png` (if VAE trained)
   - `{prefix}_clustering_analysis.png` (if GMVAE trained)

## Example Output

```
================================================================================
VAE/GMVAE SIMULATION AND TESTING
================================================================================
Using device: cpu

=== Generating Synthetic Data ===
Generated 1D dataset:
  X shape: torch.Size([1200, 100])
  True cluster centers: [-2.0, -1.0, 1.0, 2.0]
  Labels shape: torch.Size([1200])
  Z_true shape: torch.Size([1200, 1])
  Number of clusters: 4
  Points per cluster: 300

=== Training GMVAE ===
Starting training for 500 epochs...
[001] loss=7246.281  kl_w=0.02  recon=0.000  kl=0.000
[020] loss=555.706  kl_w=0.40  recon=0.000  kl=0.000
...
[500] loss=349.855  kl_w=1.00  recon=0.000  kl=0.000
GMVAE training completed!

=== Evaluating GMVAE ===
GMVAE evaluation completed:
  Latent space correlation: 0.2379
  RMSE: 1.8020
  ARI: 0.4992
  NMI: 0.6402
  Mean cluster confidence: 0.9606

================================================================================
SIMULATION RESULTS SUMMARY
================================================================================
Dataset Configuration:
  - Model type: gmvae
  - Clusters: 4
  - Latent dimension: 1
  - Features: 100
  - Points per cluster: 300
  - Total samples: 1200

GMVAE Performance:
  - Latent space correlation: 0.2379
  - RMSE: 1.8020
  - ARI: 0.4992
  - NMI: 0.6402
  - Mean cluster confidence: 0.9606

Comparison:
  ✗ GMVAE shows BETTER latent space recovery than VAE
  ✓ GMVAE shows LOWER reconstruction error than VAE
  ~ GMVAE shows MODERATE clustering performance
================================================================================
```

## Advantages over Jupyter Notebooks

1. **Reproducibility**: Deterministic execution with fixed seeds
2. **Automation**: Can be run in batch mode or as part of larger pipelines
3. **Parameterization**: Easy to sweep parameters and compare configurations
4. **Documentation**: Self-documenting with help text and examples
5. **Integration**: Easy to integrate with other tools and workflows
6. **Version Control**: Better suited for version control and collaboration

## Dependencies

The script requires the following packages:
- torch
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- scikit-learn

All dependencies should be available in the project's conda environment.

## Troubleshooting

1. **CUDA Issues**: The script automatically detects and uses available devices (CPU/CUDA/MPS)
2. **Memory Issues**: Reduce batch size or number of points per cluster
3. **Convergence Issues**: Adjust learning rate, KL warmup, or model architecture
4. **Import Errors**: Ensure the `src` directory is in the Python path

## Future Enhancements

Potential improvements for the simulation script:
1. Support for more complex data generation patterns
2. Additional model architectures (CircularVAE, etc.)
3. Hyperparameter optimization capabilities
4. Integration with experiment tracking tools
5. Support for real biological datasets
