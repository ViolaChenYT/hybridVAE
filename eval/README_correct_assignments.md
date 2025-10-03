# Correct Assignment Test

This directory contains the implementation of your advisor's request to compare the loss from your trained VAE with the loss obtained when using "correct" latent space assignments.

## What it does

The test implements the following approach:
1. **Train a VAE model** on your C. elegans data
2. **Extract lineage-specific statistics** from the trained model (mean μ_i and variance ν for each lineage)
3. **Generate "correct" assignments**: For each cell c from lineage i, sample z(c) ~ N(μ_i, ν)
4. **Compute reconstruction loss** using these correct assignments through the decoder
5. **Compare** this loss with the loss from your trained VAE

## Files

- `test_correct_assignments.py` - Main script (can be run like your existing test)
- `run_correct_assignment_test.sh` - Example bash script with different parameter combinations
- `README_correct_assignments.md` - This documentation

## Usage

### Command Line (Same as your existing test)

```bash
# Basic usage
python eval/test_correct_assignments.py \
    --h5ad_path data/ABpxp_path.h5ad \
    --label_key lineage \
    --n_epochs 100 \
    --n_warmup_epochs 20 \
    --print_every 20 \
    --n_components 5 \
    --prior_values -2 -1 0 1 2

# With batch correction
python eval/test_correct_assignments.py \
    --h5ad_path data/ABpxp_path.h5ad \
    --label_key lineage \
    --batch_correction \
    --batch_key batch \
    --n_epochs 100 \
    --n_warmup_epochs 20 \
    --print_every 20 \
    --n_components 5 \
    --prior_values -2 -1 0 1 2
```

### Using the bash script

```bash
# Make executable and run
chmod +x eval/run_correct_assignment_test.sh
./eval/run_correct_assignment_test.sh
```

### Jupyter Notebook

See `examples/test_correct_assignments.ipynb` for an interactive version with visualizations.

## Interpretation of Results

The script will output a comparison like this:

```
=== COMPARISON RESULTS ===
Trained VAE Loss: 272.3789
Correct Assignment Loss: 451.6212
Difference: 179.2422

Trained VAE Recon Loss: 272.3789
Correct Assignment Recon Loss: 451.6212
Recon Loss Difference: 179.2422

✗ Correct assignments achieve HIGHER loss (179.2422 worse)
```

### What this means:

- **If correct assignments have LOWER loss**: ✓ Your model is learning good representations! The model is finding better latent representations than simply using lineage-specific means.

- **If correct assignments have HIGHER loss**: ✗ There might be room for improvement. The model might not be learning optimal representations.

## Key Functions

- `compute_lineage_statistics()` - Extracts lineage-specific means and variances from trained model
- `generate_correct_assignments()` - Samples from lineage-specific distributions
- `compute_correct_assignment_loss()` - Computes reconstruction loss using correct assignments
- `compare_losses()` - Main function that runs the full comparison

## Parameters

The script uses the same parameters as your existing test:
- `--h5ad_path`: Path to your data file
- `--label_key`: Column name for lineage labels (default: "lineage")
- `--n_epochs`: Number of training epochs
- `--n_components`: Number of GMVAE components
- `--prior_values`: Fixed prior means for the components
- `--batch_correction`: Enable batch correction
- And more...

## Example Output

The script provides detailed output including:
- Training progress
- Lineage statistics (mean, std, number of cells per lineage)
- Loss comparisons
- Interpretation of results

This gives you a comprehensive analysis of how well your VAE is learning compared to the "correct" baseline.
