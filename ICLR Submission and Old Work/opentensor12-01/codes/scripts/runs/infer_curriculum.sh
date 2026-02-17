#!/bin/bash

# Inference script for curriculum-trained polynomial circuit model
# Tests the model on polynomials of complexity 1-6

cd /home/ubuntu/random/src/OpenTensor

python codes/scripts/polynomial_infer_curriculum.py \
    --model_path codes/scripts/runs/polynomial_net_complexity.pth \
    --n_variables 3 \
    --max_degree 3 \
    --max_nodes 10 \
    --hidden_dim 256 \
    --complexity_start 1 \
    --complexity_end 6 \
    --n_tests 20 \
    --mcts_simulations 256 \
    --use_mcts \
    --show_examples
