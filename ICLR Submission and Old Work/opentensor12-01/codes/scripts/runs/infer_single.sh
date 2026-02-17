#!/bin/bash

# Inference script for single polynomial using curriculum-trained model
# Usage:
#   bash infer_single.sh                          # Random polynomial
#   bash infer_single.sh "x0*x1 + x2"            # Specific polynomial
#   bash infer_single.sh "" 3                     # Random complexity 3
#   bash infer_single.sh "x0**2 + x1*x2"         # Another specific polynomial

cd /home/ubuntu/random/src/OpenTensor

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate binary-forms

export PYTHONPATH=/home/ubuntu/random/src/OpenTensor:/home/ubuntu/random/src:$PYTHONPATH

POLYNOMIAL="${1:-}"
COMPLEXITY="${2:-}"

if [ -z "$POLYNOMIAL" ]; then
    if [ -z "$COMPLEXITY" ]; then
        echo "Running inference on random polynomial (random complexity)..."
        python codes/scripts/polynomial_infer_single.py \
            --model_path codes/scripts/runs/polynomial_net_complexity.pth \
            --n_variables 3 \
            --max_degree 3 \
            --max_nodes 10 \
            --mcts_simulations 500
    else
        echo "Running inference on random polynomial (complexity $COMPLEXITY)..."
        python codes/scripts/polynomial_infer_single.py \
            --model_path codes/scripts/runs/polynomial_net_complexity.pth \
            --complexity "$COMPLEXITY" \
            --n_variables 3 \
            --max_degree 3 \
            --max_nodes 10 \
            --mcts_simulations 500
    fi
else
    echo "Running inference on polynomial: $POLYNOMIAL"
    python codes/scripts/polynomial_infer_single.py \
        --model_path codes/scripts/runs/polynomial_net_complexity.pth \
        --polynomial "$POLYNOMIAL" \
        --n_variables 3 \
        --max_degree 3 \
        --max_nodes 10 \
        --mcts_simulations 500
fi
