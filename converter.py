import sympy as sp
from collections import defaultdict

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

def sympy_to_vector(expr, monomial_dict=None, variable_names=None):
    """
    Convert from sympy expression to vector representation.
    
    Input:
        expr: SymPy polynomial expression
        monomial_dict (Dict[int, Tuple[int]], optional): Dictionary mapping 1-based indices to exponent tuples
                                                        If None, will be generated based on the expression
        variable_names (List[str], optional): Variable names to use (defaults to symbols in the expression)
    Output:
        vector (List[int]): Vector of coefficients
        monomial_dict (Dict[int, Tuple[int]]): Dictionary mapping 1-based indices to exponent tuples
    """
    # Convert expression to expanded polynomial form
    poly_expr = sp.expand(expr)
    
    # If variable_names is provided, create symbols from these names
    if variable_names:
        vars = sp.symbols(variable_names)
    else:
        # Otherwise extract symbols from the expression
        symbols = list(poly_expr.free_symbols)
        # Sort them to ensure consistent ordering
        symbols.sort(key=str)
        variable_names = [str(sym) for sym in symbols]
        vars = sp.symbols(variable_names)
    
    m = len(vars)
    
    # If monomial_dict is not provided, we'll create it
    if monomial_dict is None:
        # Convert to polynomial and get the degree
        if poly_expr.is_constant():
            max_degree = 0
        else:
            poly = sp.Poly(poly_expr, *vars)
            max_degree = poly.total_degree()
        
        # Generate all possible monomials up to the degree of the polynomial
        from itertools import combinations_with_replacement
        monomial_dict = {}
        index = 1  # Start with 1-based indexing
        
        for total_degree in range(max_degree + 1):
            for combo in combinations_with_replacement(range(m), total_degree):
                exponents = [0] * m
                for var_idx in combo:
                    exponents[var_idx] += 1
                monomial_dict[index] = tuple(exponents)
                index += 1
    
    # Create a map from monomial to index for quick lookup
    monomial_to_index = {monomial: idx for idx, monomial in monomial_dict.items()}
    
    # Find the highest index to determine vector size
    max_idx = max(monomial_dict.keys()) if monomial_dict else 0
    
    # Initialize the vector with zeros (using the highest index)
    vector = [0] * max_idx
    
    # Extract terms from the expanded polynomial
    if poly_expr == 0:
        return vector, monomial_dict
    
    if poly_expr.is_constant():
        # Handle the constant term
        zero_tuple = (0,) * m
        if zero_tuple in monomial_to_index:
            idx = monomial_to_index[zero_tuple]
            vector_idx = idx - 1  # Convert 1-based to 0-based
            vector[vector_idx] = int(poly_expr)
        return vector, monomial_dict
    
    # Convert to sympy polynomial
    poly = sp.Poly(poly_expr, *vars)
    
    # Extract the coefficients and monomials
    for powers, coef in poly.terms():
        # Convert to our exponent notation
        exponents = [0] * m
        for i, power in enumerate(powers):
            exponents[i] = power
        
        monomial = tuple(exponents)
        
        # Update the vector
        if monomial in monomial_to_index:
            idx = monomial_to_index[monomial]
            vector_idx = idx - 1  # Convert 1-based to 0-based
            vector[vector_idx] = coef
    
    return vector, monomial_dict

# Example usage:
if __name__ == "__main__":
    # Define variables
    x, y, z = sp.symbols('x y z')
    
    # Create a sympy expression
    expr = x**2 + 2*y + 3*z**3 + 4
    
    # Define the monomial dictionary (all possible monomials up to degree 3)
    # Now using 1-based indexing with dictionary instead of list
    monomial_dict = {
        1: (0, 0, 0),  # constant
        2: (1, 0, 0),  # x
        3: (0, 1, 0),  # y
        4: (0, 0, 1),  # z
        5: (2, 0, 0),  # x^2
        6: (1, 1, 0),  # xy
        7: (1, 0, 1),  # xz
        8: (0, 2, 0),  # y^2
        9: (0, 1, 1),  # yz
        10: (0, 0, 2),  # z^2
        11: (0, 0, 3),  # z^3
    }
    
    # Convert from sympy to vector
    vector, monomials = sympy_to_vector(expr, monomial_dict, ['x', 'y', 'z'])
    
    print("Original expression:", expr)
    print("Vector representation:", vector)
    
    # Convert back to sympy
    expr2 = vector_to_sympy(vector, monomials, ['x', 'y', 'z'])
    
    print("Converted back to sympy:", expr2)
    print("Are they equal?", sp.simplify(expr - expr2) == 0)