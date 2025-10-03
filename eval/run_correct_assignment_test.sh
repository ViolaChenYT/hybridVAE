#!/bin/bash

# Example script to run the correct assignment test
# This script demonstrates how to run the test with different parameters

echo "Running Correct Assignment Test on C. elegans data..."

# Basic run with default parameters
echo "=== Basic Test ==="
python eval/test_correct_assignments.py \
    --h5ad_path data/ABpxp_path.h5ad \
    --label_key lineage \
    --n_epochs 100 \
    --n_warmup_epochs 20 \
    --print_every 20 \
    --n_components 5 \
    --prior_values -2 -1 0 1 2

echo ""
echo "=== Test with batch correction ==="
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

echo ""
echo "=== Test with different prior values ==="
python eval/test_correct_assignments.py \
    --h5ad_path data/ABpxp_path.h5ad \
    --label_key lineage \
    --n_epochs 100 \
    --n_warmup_epochs 20 \
    --print_every 20 \
    --n_components 8 \
    --prior_values -3 -2 -1 0 1 2 3 4

echo "All tests completed!"
