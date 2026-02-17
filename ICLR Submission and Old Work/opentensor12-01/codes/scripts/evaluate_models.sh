#!/bin/bash

# Comprehensive model evaluation script
# Tests both pre-curriculum and curriculum models on various polynomial complexities

source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate binary-forms

cd /home/ubuntu/random/src/OpenTensor
export PYTHONPATH=/home/ubuntu/random/src/OpenTensor:$PYTHONPATH

echo "======================================================================"
echo "POLYNOMIAL CIRCUIT MODEL EVALUATION"
echo "======================================================================"
echo ""
echo "Testing models:"
echo "  1. polynomial_net_parallel.pth (pre-curriculum, original training)"
echo "  2. polynomial_net_curriculum.pth (curriculum learning, 300 epochs)"
echo ""

# Test specific polynomial examples of varying complexity
test_polynomials=(
    "x0"                                    # Complexity 0: single variable
    "x0 + x1"                               # Complexity 1: simple addition
    "x0 * x1"                               # Complexity 1: simple multiplication
    "x0**2 + x0 + 1"                        # Complexity 2: quadratic single var
    "x0*x1 + x0 + x1 + 1"                   # Complexity 2: mixed terms
    "(x0 + x1)**2"                          # Complexity 2: factored form
    "x0**2 + 2*x0*x1 + x1**2"               # Complexity 3: expanded (x0+x1)^2
    "x0**3 + x0**2 + x0 + 1"                # Complexity 3: cubic polynomial
    "x0**2*x1 + x0*x1**2"                   # Complexity 3: symmetric
    "(x0 + 1) * (x1 + 1)"                   # Complexity 2: factored
)

echo "======================================================================"
echo "SPECIFIC POLYNOMIAL TESTS"
echo "======================================================================"
echo ""

for model_name in "polynomial_net_parallel.pth" "polynomial_net_curriculum.pth"; do
    echo "----------------------------------------------------------------------"
    echo "Model: $model_name"
    echo "----------------------------------------------------------------------"
    
    for poly in "${test_polynomials[@]}"; do
        echo ""
        echo "Testing: $poly"
        result=$(python codes/scripts/polynomial_infer.py \
            --model_path codes/scripts/runs/$model_name \
            --polynomial "$poly" \
            --method mcts \
            --mcts_simulations 256 2>&1 | grep -E "(Success:|Total reward:|Number of steps:)")
        echo "$result"
    done
    
    echo ""
    echo "======================================================================"
done

echo ""
echo "======================================================================"
echo "RANDOM POLYNOMIAL BATCH TESTS (100 samples each)"
echo "======================================================================"
echo ""

for model_name in "polynomial_net_parallel.pth" "polynomial_net_curriculum.pth"; do
    echo "----------------------------------------------------------------------"
    echo "Model: $model_name"
    echo "----------------------------------------------------------------------"
    
    python codes/scripts/polynomial_infer.py \
        --model_path codes/scripts/runs/$model_name \
        --num_tests 100 \
        --method mcts \
        --mcts_simulations 256 2>&1 | grep -A 5 "SUMMARY"
    
    echo ""
done

echo "======================================================================"
echo "EVALUATION COMPLETE"
echo "======================================================================"
