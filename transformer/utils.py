import sympy as sp

def vector_to_sympy(vector, monomial_dict, variable_names=None):
    """
    Convert from vector representation to sympy polynomial expression.
    
    Input:
        vector (List[int]): Vector of coefficients 
                           (index i-1 corresponds to monomial at key i in monomial_dict)
        monomial_dict (Dict[int, Tuple[int]]): Dictionary mapping 1-based indices to exponent tuples
        variable_names (List[str], optional): Variable names to use (defaults to ['x0', 'x1', ..., 'xm'])

    Output:
        The polynomial as a SymPy expression
    """
    if not monomial_dict:
        return sp.S.Zero
    
    # Get the first monomial to determine the number of variables
    first_monomial = next(iter(monomial_dict.values()))
    m = len(first_monomial)  # number of variables
    
    if variable_names is None:
        variable_names = [f'x{i}' for i in range(m)]

    vars = sp.symbols(variable_names)
    expr = 0
    
    # Iterate through the monomial dictionary and pair with coefficients
    for idx, exponents in monomial_dict.items():
        # Convert 1-based index to 0-based for the vector
        vector_idx = idx - 1
        
        if vector_idx >= len(vector) or vector[vector_idx] == 0:
            continue
            
        coef = vector[vector_idx]
        term = coef
        
        for var, power in zip(vars, exponents):
            term *= var ** power
        expr += term
    
    return expr


# Helper function to encode actions with commutative operations
def encode_action(operation, node1_id, node2_id, max_nodes):
    """
    Encode an action so that commutative operations share the same action ID
    regardless of the order of input nodes.
    """
    # For commutative operations, sort the node IDs
    if operation in ["add", "multiply"]:
        node1_id, node2_id = sorted([node1_id, node2_id])
    
    pair_idx = (node1_id * (node1_id+1)) // 2 + node2_id
    
    # Final action index: pair_index * num_operations + op_index
    return pair_idx * 2 + (1 if operation == "multiply" else 0)
