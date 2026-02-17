#!/bin/bash

# Complexity-Based Adaptive Curriculum Training for Polynomial Circuits
# 
# Strategy:
# - No constant "1" term (only variables x0, x1, x2)
# - Curriculum based on circuit complexity (# of operations needed)
# - Adaptive: advances to next complexity when success rate > 70%
# - Starts at complexity 1, advances up to complexity 6

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate binary-forms

cd /home/ubuntu/random/src/OpenTensor

# Set PYTHONPATH so imports work
export PYTHONPATH=/home/ubuntu/random/src/OpenTensor:/home/ubuntu/random/src:$PYTHONPATH

python codes/scripts/polynomial_train_curriculum.py \
    --n_variables 3 \
    --max_degree 3 \
    --max_nodes 10 \
    --step_penalty -0.1 \
    --success_reward 10.0 \
    --failure_penalty -5.0 \
    --hidden_dim 256 \
    --mcts_simulations 256 \
    --c_puct 1.5 \
    --episodes_per_epoch 64 \
    --epochs 500 \
    --batch_size 256 \
    --lr 3e-4 \
    --value_coef 0.5 \
    --num_workers 32 \
    --virtual_loss 1.0 \
    --checkpoint_path None \
    --save_path codes/scripts/runs/polynomial_net_complexity.pth \
    --checkpoint_freq 50 \
    --complexity_start 1 \
    --complexity_end 6
