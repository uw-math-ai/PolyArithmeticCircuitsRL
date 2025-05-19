#!/bin/bash
# Script to train and evaluate the enhanced polynomial simplification model

# Set environment variables
export PYTHONPATH=.

# Create directory for model checkpoints
mkdir -p models

# Step 1: Train the model with supervised learning
echo "===== Starting Supervised Training ====="
python main_enhanced.py \
  --mode supervised \
  --variables 3 \
  --complexity 10 \
  --train_size 2000 \
  --epochs 100 \
  --batch_size 64 \
  --hidden_dim 256 

# Step 2: Fine-tune with reinforcement learning
echo "===== Starting Reinforcement Learning ====="
python main_enhanced.py \
  --mode reinforcement \
  --load_model best_supervised_model_n3_C10.pt \
  --rl_episodes 500 \
  --batch_size 32

# Step 3: Evaluate the model
echo "===== Evaluating Model ====="
python main_enhanced.py \
  --load_model best_rl_model_n3_C10.pt \
  --evaluate

# Step 4: Test on specific polynomials
echo "===== Testing on Specific Polynomials ====="

# Test on x^2 + 2xy + y^2
echo "Testing (x+y)^2:"
python main_enhanced.py \
  --load_model best_rl_model_n3_C10.pt \
  --polynomial "x^2+2xy+y^2" \
  --simulations 1000

# Test on x^3
echo "Testing x^3:"
python main_enhanced.py \
  --load_model best_rl_model_n3_C10.pt \
  --polynomial "x^3" \
  --simulations 1000

# Test on x*y + z
echo "Testing x*y + z:"
python main_enhanced.py \
  --load_model best_rl_model_n3_C10.pt \
  --polynomial "x*y+z" \
  --simulations 1000

echo "===== Training and Evaluation Complete ====="