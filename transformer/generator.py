import random
from itertools import combinations_with_replacement
import numpy as np

def generate_monomials_with_additive_indices(n, d):
    """
    Generate monomials with an indexing scheme where:
    index(A) + index(B) = index(A+B) when adding monomial exponents
    
    Args:
        n: Number of variables
        d: Maximum degree
        
    Returns:
        index_to_monomial, monomial_to_index, and all_monomials
    """
    # Calculate the base for our number system
    base = d + 1
    
    # Create all possible monomials up to the max degree
    all_possible_monomials = []
    for total_degree in range(d + 1):
        for combo in combinations_with_replacement(range(n), total_degree):
            exponents = [0] * n
            for var_idx in combo:
                exponents[var_idx] += 1
            all_possible_monomials.append(tuple(exponents))
    
    # Create the indexing using a number system with base (d+1)
    index_to_monomial = {}
    monomial_to_index = {}
    
    for monomial in all_possible_monomials:
        # Calculate the index: the key insight is to use a positional number system
        # where each position uses base (d+1)
        index = 0
        for i, exp in enumerate(monomial):
            # Use base^position weighting
            index += exp * (base ** i)
        
        index_to_monomial[index] = monomial
        monomial_to_index[monomial] = index
    
    # Sort by index for cleaner output
    all_monomials = [index_to_monomial[idx] for idx in sorted(index_to_monomial.keys())]
    
    return index_to_monomial, monomial_to_index, all_monomials

def create_polynomial_vector(index_to_monomial, monomial_to_index, n, var_idx=None, constant_val=None):
    """
    Create a vector representation of a polynomial (either a variable or constant)
    
    Args:
        index_to_monomial: Mapping from indices to monomials
        monomial_to_index: Mapping from monomials to indices
        n: Number of variables
        var_idx: Variable index (if creating a variable)
        constant_val: Constant value (if creating a constant)
        
    Returns:
        Vector representation of the polynomial
    """
    # Determine max vector size
    max_idx = max(monomial_to_index.values())
    vector_size = max_idx + 1
    vector = [0] * vector_size
    
    if var_idx is not None:
        # Create variable polynomial: x_var_idx
        exponents = [0] * n
        exponents[var_idx] = 1
        mono_tuple = tuple(exponents)
        
        if mono_tuple in monomial_to_index:
            idx = monomial_to_index[mono_tuple]
            vector[idx] = 1
    
    elif constant_val is not None:
        # Create constant polynomial
        zero_tuple = tuple([0] * n)
        
        if zero_tuple in monomial_to_index:
            idx = monomial_to_index[zero_tuple]
            vector[idx] = constant_val
    
    return vector

def add_polynomials_vector(poly1, poly2, mod):
    """Add two polynomial vectors"""
    # Ensure equal length
    max_len = max(len(poly1), len(poly2))
    
    if len(poly1) < max_len:
        poly1 = poly1 + [0] * (max_len - len(poly1))
    if len(poly2) < max_len:
        poly2 = poly2 + [0] * (max_len - len(poly2))

    result = [0] * max_len
    
    # Add coefficients
    for i in range(max_len):
        result[i] = (poly1[i] + poly2[i]) % mod
    
    return result

def multiply_polynomials_vector(poly1, poly2, mod):
    """Multiply two polynomial vectors using additive indexing scheme"""
    # Ensure equal length
    max_len = max(len(poly1), len(poly2))
        
    result = [0] * max_len
    
    # Multiply using the additive indexing property
    for i in range(max_len):
        if poly1[i] == 0:  # Skip zero coefficients
            continue
        for j in range(max_len):
            if poly2[j] == 0:  # Skip zero coefficients
                continue
            # With additive indexing, i+j represents the product monomial's index
            if i+j < max_len:
                result[i+j] = (result[i+j] + (poly1[i] * poly2[j])) % mod
    
    return result

def generate_random_circuit(n, d, C, mod=2):
    """
    Generate a random arithmetic circuit represented as a list of actions.
    
    Parameters:
    n - number of variables
    d - maximum degree
    C - complexity parameter (number of operations)
    mod - modulo for coefficients
    
    Returns:
    actions - list of (operation, input1_idx, input2_idx) tuples
    polynomials - list of polynomial vectors for each node
    """
    # Generate monomial indexing
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    
    # Initialize actions and polynomials lists
    actions = []
    polynomials = []
    
    # Add variable nodes (the first n nodes are variables)
    for i in range(n):
        # No action for input nodes, so use None
        actions.append(("input", None, None))
        # Create polynomial vector for variable x_i
        poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n, var_idx=i)
        polynomials.append(poly)
    
    # Add constant node (the n+1 node is constant 1)
    actions.append(("constant", None, None))
    poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n, constant_val=1)
    polynomials.append(poly)
    
    # Now add operations
    for _ in range(C):
        # Choose two input nodes (can be any previously created node)
        num_nodes = len(actions)
        input1_idx = random.randint(0, num_nodes - 1)
        input2_idx = random.randint(0, num_nodes - 1)
        
        # Choose an operation
        operation = random.choice(["add", "multiply"])
        
        # Add the action
        actions.append((operation, input1_idx, input2_idx))
        
        # Compute the result polynomial
        if operation == "add":
            poly = add_polynomials_vector(polynomials[input1_idx], polynomials[input2_idx], mod)
        else:  # multiply
            poly = multiply_polynomials_vector(polynomials[input1_idx], polynomials[input2_idx], mod)
        
        polynomials.append(poly)
    
    return actions, polynomials, index_to_monomial, monomial_to_index

def generate_random_polynomials(n, d, C, num_polynomials=10000, mod=5):
    """
    Generate random polynomials using arithmetic circuits.
    
    Parameters:
    n - number of variables
    d - maximum degree
    C - complexity parameter
    num_polynomials - number of polynomials to generate
    mod - modulo for coefficients
    
    Returns:
    index_to_monomial - dictionary mapping indices to monomials
    monomial_to_index - dictionary mapping monomials to indices 
    all_polynomials - list of polynomial vectors
    all_circuits - list of action sequences for each polynomial
    """
    all_polynomials = []
    all_circuits = []
    
    # Get monomial indexing from first call (they'll all be the same)
    first_circuit, first_polys, index_to_monomial, monomial_to_index = generate_random_circuit(n, d, C, mod)
    all_circuits.append(first_circuit)
    all_polynomials.append(first_polys[-1])  # Take the final polynomial
    
    # Generate the rest of the polynomials
    for _ in range(num_polynomials - 1):
        circuit, polys, _, _ = generate_random_circuit(n, d, C, mod)
        all_circuits.append(circuit)
        all_polynomials.append(polys[-1])  # Take the final polynomial
    
    return index_to_monomial, monomial_to_index, all_polynomials, all_circuits
