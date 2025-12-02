#!/bin/bash
# Train with "interesting" algebraic patterns (factorization focus)
# Starting from the complexity-based curriculum checkpoint

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go to src/OpenTensor (3 levels up from codes/scripts/runs)
cd "$DIR/../../.."

# Activate environment and set path
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate binary-forms
export PYTHONPATH=$(pwd):$(pwd)/..:$PYTHONPATH

nohup python -u codes/scripts/polynomial_train_interesting.py \
    --n_variables 3 \
    --max_degree 3 \
    --max_nodes 10 \
    --mcts_simulations 200 \
    --episodes_per_epoch 100 \
    --epochs 200 \
    --batch_size 256 \
    --lr 0.0001 \
    --num_workers 32 \
    --checkpoint_path "codes/scripts/runs/polynomial_net_complexity.pth" \
    --save_path "codes/scripts/runs/polynomial_net_interesting.pth" \
    --complexity_start 2 \
    --complexity_end 6 \
    > codes/scripts/runs/train_interesting.log 2>&1 &

echo "Training started! Log: codes/scripts/runs/train_interesting.log"
echo "Tail command: tail -f codes/scripts/runs/train_interesting.log"
