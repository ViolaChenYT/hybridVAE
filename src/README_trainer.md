# Training and Evaluation Framework

This document describes the unified training and evaluation framework that has been abstracted from the individual scripts into a reusable `Trainer` class.

## Overview

The `Trainer` class in `src/trainer.py` provides a comprehensive framework for training and evaluating VAE and GMVAE models. It abstracts common training procedures, evaluation metrics, and visualization functions that were previously duplicated across multiple scripts.

## Key Features

### 1. **Unified Training Loop**
- Standardized training procedures with KL annealing
- Early stopping and model checkpointing
- Component usage analysis for GMVAE models
- Configurable batch processing with batch correction support
- Gradient clipping and optimizer management

### 2. **Comprehensive Evaluation**
- Latent space recovery metrics (correlation, RMSE)
- Clustering performance metrics (ARI, NMI, confusion matrix)
- Statistical significance testing
- Component assignment confidence analysis

### 3. **Rich Visualizations**
- Training loss curves
- Encoded latent distribution histograms by true cluster
- Clustering analysis plots
- Correlation and residual analysis

### 4. **Lineage Statistics**
- Lineage-specific statistics computation
- Component distribution analysis
- Mean and variance calculations per lineage

## Usage

### Basic Usage

```python
from trainer import Trainer, set_seed

# Set random seed for reproducibility
set_seed(42)

# Initialize model
model = GMVAE(n_input=100, n_latent=2, fixed_means=fixed_means)

# Create trainer
trainer = Trainer(model, device, args, "GMVAE")

# Train the model
results = trainer.train(X, batch_index=batch_index)

# Evaluate the model
metrics = trainer.evaluate(X, labels, Z_true, batch_index=batch_index)

# Generate visualizations
trainer.plot_training_losses()
trainer.plot_encoded_latent_distribution(metrics, labels)
trainer.plot_clustering_analysis(metrics, labels)
```

### Advanced Usage

```python
# Custom training with early stopping
results = trainer.train(X, batch_index=batch_index, early_stopping=True, patience=50)

# Compute lineage statistics
lineage_stats = trainer.compute_lineage_statistics(X, labels, batch_index)

# Save plots
trainer.plot_encoded_latent_distribution(metrics, labels, save_path="results/plot.png")
```

## Integration with Existing Scripts

The framework has been integrated into all existing scripts:

### 1. **test_simulation.py**
- Uses Trainer for both VAE and GMVAE training
- Maintains all existing functionality
- Provides the same visualizations and metrics

### 2. **test_correct_assignments.py**
- Uses Trainer for GMVAE training
- Simplified lineage statistics computation
- Maintains batch correction support

### 3. **test_on_c_elegans_path.py**
- Uses Trainer for GMVAE training
- Maintains all existing functionality
- Supports batch correction

## Trainer Class Methods

### Core Methods

- `train(X, batch_index=None, early_stopping=True, patience=80)`: Train the model
- `evaluate(X, labels, Z_true=None, batch_index=None)`: Evaluate the model
- `compute_lineage_statistics(X, labels, batch_index=None)`: Compute lineage statistics

### Visualization Methods

- `plot_training_losses(save_path=None)`: Plot training loss curves
- `plot_encoded_latent_distribution(metrics, labels, save_path=None)`: Plot latent distributions
- `plot_clustering_analysis(metrics, labels, save_path=None)`: Plot clustering analysis

### Internal Methods

- `_setup_optimizer()`: Setup the optimizer
- `_compute_kl_weight(epoch)`: Compute KL weight with annealing
- `_fit_step(xb, batch_index=None, kl_weight=1.0)`: Single training step
- `_analyze_component_usage(X, batch_index=None)`: Analyze component usage

## Configuration

The Trainer class expects an `args` object with the following attributes:

```python
class Args:
    # Training parameters
    n_epochs = 300
    kl_warmup_epochs = 50
    learning_rate = 5e-4
    weight_decay = 1e-5
    batch_size = 128
    print_every = 20
    
    # Model parameters
    prior_weight = 1.0  # For KL weight scaling
    
    # GMVAE stabilization parameters
    entropy_warmup_epochs = 20  # Epochs to keep entropy bonus
    lambda_entropy = 1e-3       # Weight for entropy bonus
```

## GMVAE Component Usage Stabilization

The Trainer class includes advanced stabilization techniques for GMVAE models to prevent component collapse and ensure uniform component usage:

### **Key Features**

1. **Split KL Divergence**: Separate control of categorical (β_c) and Gaussian (β_z) KL terms
2. **Stabilized Training**: β_c=1.0 from start, β_z warms up gradually
3. **Entropy Bonus**: Optional entropy regularization during early epochs
4. **Uniform Initialization**: Zero-initialized categorical encoder for uniform start
5. **Usage Monitoring**: Real-time component usage analysis

### **How It Works**

```python
# The stabilization automatically applies to GMVAE models
trainer = Trainer(gmvae_model, device, args, "GMVAE")
results = trainer.train(X)

# You'll see output like:
# === Component Usage Analysis (Epoch 0) ===
# Usage: [0.25 0.25 0.25 0.25]  # Perfectly uniform
# Mean entropy: 1.386
# Log K: 1.386
# Max deviation from uniform: 0.000
```

### **Training Progress Example**

```
[001] loss=9688.028  kl_w=0.00  recon=9687.602  kl=207165.039
    β_c=1.00, β_z=0.00  # Categorical KL on, Gaussian KL off

[010] loss=2211.974  kl_w=0.18  recon=1992.530  kl=1215.953
    β_c=1.00, β_z=0.18  # Gaussian KL gradually increasing

=== Component Usage Analysis (Epoch 10) ===
Usage: [0.658 0.137 0.085 0.121]  # Some specialization happening
Mean entropy: 0.688
Max deviation from uniform: 0.408
```

### **Command Line Arguments**

```bash
# Basic usage with default stabilization
python test_simulation.py --model_type gmvae --n_clusters 4

# Custom entropy bonus settings
python test_simulation.py --model_type gmvae --n_clusters 4 \
    --entropy_warmup_epochs 30 \
    --lambda_entropy 2e-3

# Disable entropy bonus
python test_simulation.py --model_type gmvae --n_clusters 4 \
    --entropy_warmup_epochs 0
```

### **Why This Works**

1. **Zero-initialized categorical encoder** → uniform responsibilities at t=0
2. **β_c = 1.0 anchors q(c|x)** to uniform prior early, preventing single component dominance
3. **Entropy bonus** adds inertia against premature sharpening
4. **Gradual β_z warmup** allows decoder to learn before Gaussian KL dominates
5. **Real-time monitoring** helps identify and debug usage issues

### **Expected Behavior**

- **Epoch 0**: Usage ≈ uniform, entropy ≈ log K
- **Early epochs**: Remains near uniform (β_c=1.0 + entropy bonus)
- **Later epochs**: Gradually adapts as β_z ramps and decoder learns
- **Final**: Balanced usage with good reconstruction and clustering

## Benefits

### 1. **Code Reusability**
- Eliminates duplicate training code across scripts
- Consistent training procedures
- Easy to maintain and update

### 2. **Standardization**
- Uniform evaluation metrics
- Consistent visualization styles
- Standardized logging and progress reporting

### 3. **Extensibility**
- Easy to add new evaluation metrics
- Simple to extend with new visualization types
- Modular design allows for easy customization

### 4. **Reproducibility**
- Built-in seed management
- Deterministic training procedures
- Consistent random number generation

## Migration Guide

### Before (Old Approach)
```python
# Each script had its own training function
def train_model(model, X, args, model_name="Model"):
    # 50+ lines of training code
    # Duplicated across multiple scripts
    pass

def evaluate_model(model, X, labels, Z_true, model_name="Model"):
    # 40+ lines of evaluation code
    # Duplicated across multiple scripts
    pass
```

### After (New Approach)
```python
# Single, reusable Trainer class
trainer = Trainer(model, device, args, "Model")
results = trainer.train(X)
metrics = trainer.evaluate(X, labels, Z_true)
```

## Future Enhancements

Potential improvements for the training framework:

1. **Hyperparameter Optimization**: Integration with Optuna or similar tools
2. **Experiment Tracking**: Integration with Weights & Biases or MLflow
3. **Distributed Training**: Support for multi-GPU training
4. **Advanced Scheduling**: Learning rate scheduling and warmup strategies
5. **Model Checkpointing**: More sophisticated checkpoint management
6. **Custom Loss Functions**: Support for custom loss functions
7. **Validation Sets**: Built-in validation and early stopping

## Dependencies

The Trainer class requires the following packages:
- torch
- numpy
- matplotlib
- seaborn
- pandas
- scipy
- scikit-learn

All dependencies are included in the project's conda environment.
