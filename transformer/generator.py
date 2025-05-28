import random
from itertools import combinations_with_replacement
import numpy as np

def generate_monomials_with_additive_indices(n, d):
    """
    Generate monomials with an indexing scheme where:
    index(A) + index(B) = index(A+B) when adding monomial exponents.

    Args:
        n: Number of variables
        d: Maximum degree

    Returns:
        index_to_monomial, monomial_to_index, and all_monomials
    """
    base = d + 1
    all_possible_monomials = []
    for total_degree in range(d + 1):
        for combo in combinations_with_replacement(range(n), total_degree):
            exponents = [0] * n
            for var_idx in combo:
                exponents[var_idx] += 1
            all_possible_monomials.append(tuple(exponents))

    index_to_monomial = {}
    monomial_to_index = {}
    for monomial in all_possible_monomials:
        index = 0
        for i, exp in enumerate(monomial):
            index += exp * (base ** i)
        index_to_monomial[index] = monomial
        monomial_to_index[monomial] = index

    all_monomials = [index_to_monomial[idx] for idx in sorted(index_to_monomial.keys())]
    return index_to_monomial, monomial_to_index, all_monomials

def create_polynomial_vector(index_to_monomial, monomial_to_index, n, d, var_idx=None, constant_val=None):
    """
    Create a vector representation of a polynomial using additive indexing.
    Note: The size depends on n and d via the monomial indexing.
    """
    base = d + 1
    max_idx = 0
    # Calculate max_idx based on max degree d and n variables
    # The highest index corresponds to x_(n-1)^d
    for i in range(n):
      max_idx += d * (base ** i)

    vector_size = max_idx + 1
    vector = [0] * vector_size

    if var_idx is not None:
        exponents = [0] * n
        exponents[var_idx] = 1
        mono_tuple = tuple(exponents)
        if mono_tuple in monomial_to_index:
            idx = monomial_to_index[mono_tuple]
            if idx < vector_size: vector[idx] = 1
    elif constant_val is not None:
        zero_tuple = tuple([0] * n)
        if zero_tuple in monomial_to_index:
            idx = monomial_to_index[zero_tuple]
            if idx < vector_size: vector[idx] = constant_val

    return vector

def add_polynomials_vector(poly1, poly2, mod):
    """Add two polynomial vectors."""
    max_len = max(len(poly1), len(poly2))
    poly1 = poly1 + [0] * (max_len - len(poly1))
    poly2 = poly2 + [0] * (max_len - len(poly2))
    result = [(p1 + p2) % mod for p1, p2 in zip(poly1, poly2)]
    return result

def multiply_polynomials_vector(poly1, poly2, mod, index_to_monomial, n, d):
    """Multiply two polynomial vectors using additive indexing."""
    base = d + 1
    max_idx = 0
    for i in range(n):
        max_idx += d * (base ** i)
    vector_size = max_idx + 1

    result = [0] * vector_size

    for i in range(len(poly1)):
        if poly1[i] == 0: continue
        for j in range(len(poly2)):
            if poly2[j] == 0: continue

            # Check if the resulting index is within bounds before adding
            if i + j < vector_size:
                result[i + j] = (result[i + j] + (poly1[i] * poly2[j])) % mod

    return result

def generate_random_circuit(n, d, C, mod=2):
    """
    Generate a random arithmetic circuit represented as a list of actions.
    Now allows multiplication by constants.
    """
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    actions = []
    polynomials = []

    # Add variable nodes
    for i in range(n):
        actions.append(("input", None, None))
        poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n, d, var_idx=i)
        polynomials.append(poly)

    # Add constant node
    actions.append(("constant", None, None))
    poly = create_polynomial_vector(index_to_monomial, monomial_to_index, n, d, constant_val=1)
    polynomials.append(poly)

    # Add operations
    for _ in range(C):
        num_nodes = len(actions)
        input1_idx = random.randint(0, num_nodes - 1)
        input2_idx = random.randint(0, num_nodes - 1)
        operation = random.choice(["add", "multiply"])

        # *** MODIFICATION: Removed the restriction on multiplying by constants ***
        # No need for the while loop that prevented multiplication by node 'n'.

        actions.append((operation, input1_idx, input2_idx))

        if operation == "add":
            poly = add_polynomials_vector(polynomials[input1_idx], polynomials[input2_idx], mod)
        else: # multiply
            poly = multiply_polynomials_vector(polynomials[input1_idx], polynomials[input2_idx], mod, index_to_monomial, n, d)

        polynomials.append(poly)

    # Trim unused operations (optional but good for clean data)
    actions, polynomials, _ = trim_circuit(actions, polynomials)

    return actions, polynomials, index_to_monomial, monomial_to_index

def trim_circuit(actions, polynomials):
    """
    Trim unused operations while preserving all input and constant nodes.
    """
    used = set()
    num_inputs_constants = 0
    for i, (op, _, _) in enumerate(actions):
        if op in ("input", "constant"):
            used.add(i)
            num_inputs_constants +=1

    # If only inputs/constants exist, return as is.
    if len(actions) == num_inputs_constants:
        return actions, polynomials, {i: i for i in range(len(actions))}

    stack = [len(actions) - 1] # Start from final node
    visited_in_stack = set() # Prevent cycles in stack processing

    while stack:
        idx = stack.pop()
        if idx in used or idx in visited_in_stack:
            continue

        used.add(idx)
        visited_in_stack.add(idx) # Mark as visited for this trace
        op, in1, in2 = actions[idx]
        if op in ("add", "multiply"):
            if in1 is not None:
                stack.append(in1)
            if in2 is not None:
                stack.append(in2)

    used = sorted(list(used)) # Ensure consistent order
    remap = {old: new for new, old in enumerate(used)}

    new_actions = []
    new_polynomials = []
    for old_idx in used:
        op, in1, in2 = actions[old_idx]
        if op in ("add", "multiply"):
             # Ensure inputs were kept, otherwise this node is invalid
            if in1 in remap and in2 in remap:
                new_actions.append((op, remap[in1], remap[in2]))
                new_polynomials.append(polynomials[old_idx])
            else: # This node became invalid due to trimming, try to recover or drop
                 # For simplicity, we drop, but a more complex logic could try to fix.
                 # Re-calculate 'used' and 'remap' if dropping many nodes.
                 # Here, we assume a mostly valid structure or accept some loss.
                 pass # Or handle more gracefully
        else:
            new_actions.append((op, None, None))
            new_polynomials.append(polynomials[old_idx])

    # Re-calculate remap based on potentially dropped nodes
    final_used_indices = [idx for idx, (op, _, _) in enumerate(new_actions)]
    remap = {old: new for new, old in enumerate(final_used_indices)} # This might need a rethink based on how new_actions is built

    # A simpler approach: Just filter and remap once
    used = sorted(list(used))
    remap = {old: new for new, old in enumerate(used)}
    new_actions_final = []
    new_polynomials_final = []
    for old_idx in used:
        op, in1, in2 = actions[old_idx]
        if op in ("add", "multiply"):
            new_actions_final.append((op, remap[in1], remap[in2]))
        else:
            new_actions_final.append((op, None, None))
        new_polynomials_final.append(polynomials[old_idx])


    return new_actions_final, new_polynomials_final, remap




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
        circuit, polys, _ = trim_circuit(circuit, polys)
        all_circuits.append(circuit)
        all_polynomials.append(polys[-1])  # Take the final polynomial
    
    return index_to_monomial, monomial_to_index, all_polynomials, all_circuits