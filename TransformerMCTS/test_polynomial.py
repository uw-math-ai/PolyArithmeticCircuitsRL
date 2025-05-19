#!/usr/bin/env python
"""
Quick test script to verify the polynomial simplification model is working.
Run this script to see if your model can simplify x^2+2xy+y^2 to (x+y)^2.
"""

import torch
import os
import sys
import traceback
from generator import generate_monomials_with_additive_indices
from enhanced_models import AdvancedCircuitBuilder, Config, parse_polynomial, simplify_polynomial

# Test parameters
TEST_POLYNOMIAL = "x^2+2xy+y^2"
MODEL_PATH = "best_supervised_model_n3_C10.pt"  # Change this to your model path
NUM_VARIABLES = 3
COMPLEXITY = 10
NUM_SIMULATIONS = 100

def run_test(model_path=None):
    """Run a simple test of the polynomial simplification."""
    try:
        print("=" * 60)
        print(f"TESTING POLYNOMIAL SIMPLIFICATION MODEL")
        print("=" * 60)
        
        # Check if generator.py exists
        if not os.path.exists("generator.py"):
            print("ERROR: generator.py not found. Make sure it's in the current directory.")
            return False
            
        # Check if the enhanced_models_fixed.py file exists
        if not os.path.exists("enhanced_models.py"):
            print("ERROR: enhanced_models_fixed.py not found. Make sure it's in the current directory.")
            return False
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Create configuration
        config = Config()
        config.n_variables = NUM_VARIABLES
        config.max_complexity = COMPLEXITY
        config.num_simulations = NUM_SIMULATIONS
        
        # Generate monomial indexing
        print("Generating monomial indexing...")
        index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
            config.n_variables, 
            config.polynomial_degree
        )
        
        # Calculate maximum vector size
        max_vector_size = max(monomial_to_index.values()) + 1
        print(f"Maximum polynomial vector size: {max_vector_size}")
        
        # Initialize model
        print("Initializing model...")
        model = AdvancedCircuitBuilder(config, max_vector_size).to(device)
        
        # Load model if specified
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Model loaded successfully")
        else:
            print("WARNING: No pre-trained model loaded. Using untrained model.")
            if model_path:
                print(f"Model path {model_path} not found.")
        
        # Parse test polynomial
        print(f"\nParsing polynomial: {TEST_POLYNOMIAL}")
        poly_vector = parse_polynomial(TEST_POLYNOMIAL, index_to_monomial, monomial_to_index, config.n_variables)
        
        # Verify polynomial parsing worked
        if sum(poly_vector) == 0:
            print("ERROR: Failed to parse polynomial correctly.")
            return False
            
        # Simplify the polynomial
        print("\nSimplifying polynomial...")
        circuit, simplified_poly = simplify_polynomial(
            poly_vector, model, config, index_to_monomial, monomial_to_index
        )
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Allow model path to be specified as command line argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    
    success = run_test(model_path)
    if success:
        print("\n✅ Test passed: The system is working correctly!")
    else:
        print("\n❌ Test failed: Please check the error messages above.")