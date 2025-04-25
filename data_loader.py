import random
import json
import numpy as np
import pickle
import os
from itertools import combinations_with_replacement
import sympy as sp
from converter import vector_to_sympy, sympy_to_vector

# [Previous code functions here - generate_random_polynomials, etc.]

def save_polynomial_data(index_to_monomial, polynomials, filename_prefix, format="numpy"):
    """
    Save the polynomial data to disk.
    
    Parameters:
    index_to_monomial - dictionary mapping indices to monomials
    polynomials - list of polynomial vectors
    filename_prefix - prefix for the saved files
    format - "numpy" or "pickle" (default: "numpy")
    """
    if format == "numpy":
        # Convert polynomials to numpy array for efficient storage
        poly_array = np.array(polynomials, dtype=np.int8)
        
        # Save the polynomial vectors as a numpy array
        np.save(f"{filename_prefix}_polynomials.npy", poly_array)
        
        # Save the index_to_monomial dictionary as a JSON file
        # Convert tuple keys to strings for JSON compatibility
        monomial_dict_str = {str(idx): list(monomial) for idx, monomial in index_to_monomial.items()}
        
        with open(f"{filename_prefix}_monomials.json", "w") as f:
            json.dump(monomial_dict_str, f)
            
        print(f"Saved {len(polynomials)} polynomials in NumPy/JSON format with prefix '{filename_prefix}'")
        print(f"Files created: {filename_prefix}_polynomials.npy, {filename_prefix}_monomials.json")
    
    elif format == "pickle":
        # Save everything in a single pickle file
        data = {
            "index_to_monomial": index_to_monomial,
            "polynomials": polynomials
        }
        
        with open(f"{filename_prefix}.pkl", "wb") as f:
            pickle.dump(data, f)
            
        print(f"Saved {len(polynomials)} polynomials in pickle format: {filename_prefix}.pkl")
    
    else:
        raise ValueError("Format must be 'numpy' or 'pickle'")

def load_polynomial_data(filename_prefix, format="numpy"):
    """
    Load the polynomial data from disk.
    
    Parameters:
    filename_prefix - prefix for the saved files
    format - "numpy" or "pickle" (default: "numpy")
    
    Returns:
    index_to_monomial - dictionary mapping indices to monomials
    polynomials - list of polynomial vectors
    """
    if format == "numpy":
        # Load the polynomial vectors
        polynomials = np.load(f"{filename_prefix}_polynomials.npy").tolist()
        
        # Load the index_to_monomial dictionary
        with open(f"{filename_prefix}_monomials.json", "r") as f:
            monomial_dict_str = json.load(f)
        
        # Convert string keys back to integers and string values back to tuples
        index_to_monomial = {int(idx): tuple(monomial) for idx, monomial in monomial_dict_str.items()}
        
        print(f"Loaded {len(polynomials)} polynomials from NumPy/JSON format with prefix '{filename_prefix}'")
    
    elif format == "pickle":
        # Load everything from the pickle file
        with open(f"{filename_prefix}.pkl", "rb") as f:
            data = pickle.load(f)
        
        index_to_monomial = data["index_to_monomial"]
        polynomials = data["polynomials"]
        
        print(f"Loaded {len(polynomials)} polynomials from pickle format: {filename_prefix}.pkl")
    
    else:
        raise ValueError("Format must be 'numpy' or 'pickle'")
    
    return index_to_monomial, polynomials

def prepare_training_data(index_to_monomial, polynomials, n, m, C):
    """
    Prepare training data for a reinforcement learning algorithm.
    
    Parameters:
    index_to_monomial - dictionary mapping indices to monomials
    polynomials - list of polynomial vectors
    n - maximum degree
    m - number of variables
    C - complexity parameter
    
    Returns:
    Dictionary with training data structured for RL
    """
    # Create variable names for easier readability
    var_names = [f'x{i}' for i in range(m)]
    
    # Convert index_to_monomial to readable form for debugging
    readable_monomials = {}
    for idx, monomial in index_to_monomial.items():
        term = []
        for i, power in enumerate(monomial):
            if power > 0:
                if power == 1:
                    term.append(f'{var_names[i]}')
                else:
                    term.append(f'{var_names[i]}^{power}')
        
        if not term:
            readable_monomials[idx] = "1"
        else:
            readable_monomials[idx] = "*".join(term)
    
    # Structure data for RL training
    training_data = {
        "metadata": {
            "n": n,
            "m": m,
            "C": C,
            "num_polynomials": len(polynomials),
            "num_monomials": len(index_to_monomial)
        },
        "index_to_monomial": index_to_monomial,
        "readable_monomials": readable_monomials,
        "polynomials": polynomials
    }
    
    return training_data

# Example usage
if __name__ == "__main__":
    # Parameters for polynomial generation
    n = 5  # maximum degree
    m = 6  # number of variables
    C = 4  # complexity parameter
    i = 2  # degree of first polynomial
    num_samples = 1000  # number of polynomials to generate
    
    # Generate the polynomials
    print(f"Generating {num_samples} random polynomials...")
    index_to_monomial, polynomials = generate_random_polynomials(n, m, C, i, num_polynomials=num_samples)
    
    # Save the data in both formats
    save_polynomial_data(index_to_monomial, polynomials, "poly_data", format="numpy")
    save_polynomial_data(index_to_monomial, polynomials, "poly_data", format="pickle")
    
    # Load the data (for example purposes)
    loaded_index_to_monomial, loaded_polynomials = load_polynomial_data("poly_data", format="numpy")
    
    # Prepare training data
    training_data = prepare_training_data(loaded_index_to_monomial, loaded_polynomials, n, m, C)
    
    # Print example of the prepared data
    print("\nPrepared training data:")
    print(f"Number of polynomials: {training_data['metadata']['num_polynomials']}")
    print(f"Number of monomials: {training_data['metadata']['num_monomials']}")
    
    # Show a sample polynomial
    sample_idx = 0
    print(f"\nSample polynomial (vector form):")
    print(polynomials[sample_idx][:10], "... (truncated)")
    
    # Convert to readable form
    non_zero_terms = []
    for idx, coef in enumerate(polynomials[sample_idx]):
        if coef != 0:
            monomial_str = training_data['readable_monomials'][idx]
            non_zero_terms.append(monomial_str)
    
    print("Sample polynomial (readable form):")
    print(" + ".join(non_zero_terms))