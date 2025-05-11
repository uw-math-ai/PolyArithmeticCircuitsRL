import random
from itertools import combinations_with_replacement
import sympy as sp
from converter import vector_to_sympy, sympy_to_vector

# Define a class to represent nodes in our arithmetic circuit
class CircuitNode:
    def __init__(self, node_type, inputs=None, value=None, operation=None):
        self.node_type = node_type  # "input", "constant", "add", or "multiply"
        self.inputs = inputs or []  # List of input nodes
        self.value = value          # Used for constants and input variables
        self.operation = operation  # Operation type ("add" or "multiply")
    
    def __repr__(self):
        if self.node_type == "input":
            return f"Input(x_{self.value})"
        elif self.node_type == "constant":
            return f"Constant({self.value})"
        elif self.node_type == "operation":
            return f"Operation({self.operation}, inputs={len(self.inputs)})"
        return f"Node({self.node_type})"

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

def generate_random_polynomials(n, d, C, num_polynomials=10000, mod=5):
    """
    Generate random polynomials by multiplying polynomials of specific degrees.
    
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
    all_circuits - list of arithmetic circuits for each polynomial
    """
    index_to_monomial, monomial_to_index, all_monomials = generate_monomials_with_additive_indices(n, d)

    all_polynomials = []
    all_circuits = []  # Store the arithmetic circuits
    
    for _ in range(num_polynomials):
        # Generate polynomial of complexity C
        i = random.randint(1, C-1)
        operation = random.choice(["add", "multiply"])

        first_poly, first_circuit = generate_random_polynomials_recursive(n, d, i, mod)
        second_poly, second_circuit = generate_random_polynomials_recursive(n, d, C-i, mod)

        # Create a circuit node for this operation
        circuit_node = CircuitNode(
            node_type="operation",
            inputs=[first_circuit, second_circuit],
            operation=operation
        )

        if operation == 'add':
            poly_dict = add_polynomials(first_poly, second_poly, d, mod)
        else:
            poly_dict = multiply_polynomials(first_poly, second_poly, d, mod)
        
        # Convert to vector
        maxim = 0
        for ind in index_to_monomial:
            if ind > maxim:
                maxim = ind
        
        vector = [0] * (maxim+1)
        for monomial, coef in poly_dict.items():
            if monomial in monomial_to_index:
                vector[monomial_to_index[monomial]] = coef
        
        all_polynomials.append(vector)
        all_circuits.append(circuit_node)
    
    return index_to_monomial, monomial_to_index, all_polynomials, all_circuits

def generate_random_polynomials_recursive(n, d, C, mod):
    """
    Recursively generate random polynomials by multiplying polynomials of specific degrees.
    
    Parameters:
    n - number of variables
    d - maximum degree
    C - complexity parameter
    mod - modulo for coefficients
    
    Returns:
    poly_dict - polynomial as a dictionary
    circuit - arithmetic circuit representing the polynomial
    """
    # Base case: Generate a single variable or constant
    if C == 1:
        var_idx = random.randint(0, n)
        exponents = [0] * n
        
        # Create circuit node
        if var_idx == n:
            # Constant term
            circuit = CircuitNode(node_type="constant", value=1)
        else:
            # Variable term
            exponents[var_idx] = 1
            circuit = CircuitNode(node_type="input", value=var_idx)
            
        poly = {tuple(exponents): 1}
        return poly, circuit
    else:
        # Recursive case: Generate composite polynomial
        i = random.randint(1, C-1)
        operation = random.choice(["add", "multiply"])

        first_poly, first_circuit = generate_random_polynomials_recursive(n, d, i, mod)
        second_poly, second_circuit = generate_random_polynomials_recursive(n, d, C-i, mod)

        # Create circuit node for this operation
        circuit = CircuitNode(
            node_type="operation",
            inputs=[first_circuit, second_circuit],
            operation=operation
        )

        if operation == 'add':
            poly_dict = add_polynomials(first_poly, second_poly, d, mod)
        else:
            poly_dict = multiply_polynomials(first_poly, second_poly, d, mod)

        return poly_dict, circuit

def add_polynomials(poly1, poly2, d, mod):
    """Add two polynomials, restricting to terms of degree <= d"""
    result = poly1.copy()
    
    for monomial, coef in poly2.items():
        if sum(monomial) > d:
            continue  # Skip terms with degree > d
        
        if monomial in result:
            result[monomial] = (result[monomial] + coef) % mod
            if result[monomial] == 0:
                del result[monomial]
        else:
            result[monomial] = coef
    
    return result

def multiply_polynomials(poly1, poly2, d, mod):
    """Multiply two polynomials, restricting to terms of degree <= d"""
    result = {}
    
    for mon1, coef1 in poly1.items():
        for mon2, coef2 in poly2.items():
            # Add the exponents
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            
            # Skip terms with degree > d
            if sum(new_mon) > d:
                continue
            
            # Multiply the coefficients
            new_coef = (coef1 * coef2) % mod
            
            # Add to the result
            if new_mon in result:
                result[new_mon] = (result[new_mon] + new_coef) % mod
                if result[new_mon] == 0:
                    del result[new_mon]
            else:
                if new_coef != 0:
                    result[new_mon] = new_coef
    
    return result

def add_polynomials_vector(poly1, poly2, mod):
    """Add two polynomial vectors"""
    if len(poly1) != len(poly2):
        return "error"

    max_len = max(len(poly1), len(poly2))
    
    result = [0] * max_len
    
    # Copy poly1
    for i in range(len(poly1)):
        result[i] = poly1[i]
    
    # Add poly2
    for i in range(len(poly2)):
        result[i] = (result[i] + poly2[i]) % mod
    
    return result

def multiply_polynomials_vector(poly1, poly2, mod, index_to_monomial, monomial_to_index):
    """
    Multiply two polynomial vectors using the additive indexing scheme
    """
    if len(poly1) != len(poly2):
        return "error"

    result = [0] * len(poly1)
    
    for i in range(len(poly1)):
        if poly1[i] == 0:  # Skip zero coefficients
            continue
            
        for j in range(len(poly2)):
            if poly2[j] == 0:  # Skip zero coefficients
                continue
            
            # With additive indexing, i+j represents the product monomial's index
            if i+j < len(result):
                result[i+j] = (result[i+j] + (poly1[i] * poly2[j])) % mod
    
    return result
