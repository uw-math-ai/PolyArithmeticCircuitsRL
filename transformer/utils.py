import sympy as sp

import sympy as sp


def vector_to_sympy(vector, index_to_monomial, variable_names=None):
    """
    Convert from vector representation to sympy polynomial expression.

    Input:
        vector (torch.Tensor or List[float]): Vector of coefficients
        index_to_monomial (Dict[int, Tuple[int]]): Dictionary mapping indices to exponent tuples
        variable_names (List[str], optional): Variable names to use (defaults to ['x0', 'x1', ..., 'xn'])

    Output:
        The polynomial as a SymPy expression
    """
    # Convert tensor to list if needed
    if hasattr(vector, "cpu"):
        vector = vector.cpu().numpy().tolist()
    elif hasattr(vector, "numpy"):
        vector = vector.numpy().tolist()
    elif hasattr(vector, "tolist"):
        vector = vector.tolist()

    # Get the first monomial to determine the number of variables
    if not index_to_monomial:
        return sp.S.Zero

    first_monomial = next(iter(index_to_monomial.values()))
    n = len(first_monomial)  # number of variables

    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n)]

    vars = sp.symbols(variable_names)
    expr = 0

    # Iterate through the vector indices
    for idx in range(len(vector)):
        if idx in index_to_monomial and vector[idx] != 0:
            coef = vector[idx]
            exponents = index_to_monomial[idx]
            term = coef

            for var, power in zip(vars, exponents):
                if power > 0:  # Skip variables with zero power
                    term *= var**power
            expr += term

    return expr


# Helper function to encode actions with commutative operations
def encode_action(operation, node1_id, node2_id, max_nodes):
    """
    Encode an action so that commutative operations share the same action ID
    regardless of the order of input nodes.
    """
    # For commutative operations, sort the node IDs
    # if operation in ["add", "multiply"]:
    #     node1_id, node2_id = sorted([node1_id, node2_id])
    if node1_id > node2_id:
        node1_id, node2_id = node2_id, node1_id

    pair_idx = (node1_id * (node1_id + 1)) // 2 + node2_id

    # Final action index: pair_index * num_operations + op_index
    return pair_idx * 2 + (1 if operation == "multiply" else 0)
