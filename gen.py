import random
from itertools import combinations_with_replacement
import sympy as sp
from converter import vector_to_sympy, sympy_to_vector


def generate_random_polynomials(n, m, C, i, num_polynomials=10000):
    """
    Generate random polynomials by multiplying polynomials of specific degrees.
    
    Parameters:
    n - maximum degree
    m - number of variables
    C - complexity parameter
    i - degree of first polynomial
    num_polynomials - number of polynomials to generate
    
    Returns:
    index_to_monomial - dictionary mapping indices to monomials
    all_polynomials - list of polynomial vectors
    """
    if i > C:
        return "error"
    
    # Generate all possible monomials up to degree n
    all_monomials = []
    for total_degree in range(n + 1):
        for combo in combinations_with_replacement(range(m), total_degree):
            exponents = [0] * m
            for var_idx in combo:
                exponents[var_idx] += 1
            
            all_monomials.append(tuple(exponents))
    
    all_monomials.sort()
    
    # Create index_to_monomial dictionary (mapping indices to monomials)
    index_to_monomial = {idx: monomial for idx, monomial in enumerate(all_monomials)}
    
    # Also create the reverse mapping for internal use
    monomial_to_index = {monomial: idx for idx, monomial in enumerate(all_monomials)}

    all_polynomials = []
    
    for _ in range(num_polynomials):
        # Generate first polynomial of degree i
        first_poly = generate_random_polynomial(n, m, i)
                
        # Generate second polynomial of degree C-1-i
        second_poly = generate_random_polynomial(n, m, C-1-i)
                
        # Convert back to dictionary form
        poly_dict = multiply_polynomials(first_poly, second_poly, n)
        
        # Convert to vector
        vector = [0] * len(all_monomials)
        for monomial, coef in poly_dict.items():
            if monomial in monomial_to_index:
                vector[monomial_to_index[monomial]] = coef
        
        all_polynomials.append(vector)
    
    return index_to_monomial, all_polynomials

def generate_random_polynomial(n, m, C):
    """
    Generate a random polynomial with specific constraints.
    
    Parameters:
    n - maximum degree
    m - number of variables
    C - complexity parameter (number of operations)
    
    Returns:
    A dictionary representing the polynomial
    """
    if random.choice([True, False]):
        # Start with 1
        poly = {(0,) * m: 1}
    else:
        # Start with a single variable
        var_idx = random.randint(0, m - 1)
        exponents = [0] * m
        exponents[var_idx] = 1
        poly = {tuple(exponents): 1}
    
    # Perform C successive operations
    for _ in range(C):
        # Choose between addition and multiplication
        operation = random.choice(["add", "multiply"])
        
        # Generate a simple term
        if random.choice([True, False]):
            # Use 1
            new_poly = {(0,) * m: 1}
        else:
            # Use a single variable
            var_idx = random.randint(0, m - 1)
            exponents = [0] * m
            exponents[var_idx] = 1
            new_poly = {tuple(exponents): 1}
        
        if operation == "add":
            poly = add_polynomials(poly, new_poly, n)
        else:  # multiply
            poly = multiply_polynomials(poly, new_poly, n)
    
    return poly

def add_polynomials(poly1, poly2, n):
    """Add two polynomials, restricting to terms of degree <= n"""
    result = poly1.copy()
    
    for monomial, coef in poly2.items():
        if sum(monomial) > n:
            continue  # Skip terms with degree > n
        
        if monomial in result:
            result[monomial] = (result[monomial] + coef) % 2 #mod2
            if result[monomial] == 0:
                del result[monomial]
        else:
            result[monomial] = coef
    
    return result

def multiply_polynomials(poly1, poly2, n):
    """Multiply two polynomials, restricting to terms of degree <= n"""
    result = {}
    
    for mon1, coef1 in poly1.items():
        for mon2, coef2 in poly2.items():
            # Add the exponents
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            
            # Skip terms with degree > n
            if sum(new_mon) > n:
                continue
            
            # Multiply the coefficients mod2
            new_coef = (coef1 * coef2) % 2
            
            # Add to the result
            if new_mon in result:
                result[new_mon] = (result[new_mon] + new_coef) % 2
                if result[new_mon] == 0:
                    del result[new_mon]
            else:
                result[new_mon] = new_coef
    
    return result

# Example usage
if __name__ == "__main__":
    index_to_monomial, polynomials = generate_random_polynomials(2, 2, 4, 2, num_polynomials=20)
    print(f"Generated {len(polynomials)} polynomials")
    for i in (polynomials):
        print(f"polynomial: {i}")

    print(f"Index to monomial mapping:")
    for idx, monomial in list(index_to_monomial.items()):
        print(f"  {idx}: {monomial}")
    print("  ...")