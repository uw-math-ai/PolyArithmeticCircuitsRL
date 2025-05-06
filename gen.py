import random
from itertools import combinations_with_replacement
import sympy as sp
from converter import vector_to_sympy, sympy_to_vector

def generate_monomials(n,d):
    all_monomials = []
    for total_degree in range(d + 1):
        for combo in combinations_with_replacement(range(n), total_degree):
            exponents = [0] * n
            for var_idx in combo:
                exponents[var_idx] += 1
            
            all_monomials.append(tuple(exponents))
    
    all_monomials.sort()
    
    # Create index_to_monomial dictionary (mapping indices to monomials)
    index_to_monomial = {idx: monomial for idx, monomial in enumerate(all_monomials)}
    
    # Also create the reverse mapping for internal use
    monomial_to_index = {monomial: idx for idx, monomial in enumerate(all_monomials)}
    return index_to_monomial, monomial_to_index, all_monomials


def generate_random_polynomials(n, d, C, num_polynomials=10000, mod=5):
    """
    Generate random polynomials by multiplying polynomials of specific degrees.
    
    Parameters:
    n - number of variables
    d - maximum degree
    C - complexity parameter
    num_polynomials - number of polynomials to generate
    
    Returns:
    index_to_monomial - dictionary mapping indices to monomials
    all_polynomials - list of polynomial vectors
    """

    index_to_monomial, monomial_to_index, all_monomials = generate_monomials(n,d)

    all_polynomials = []
    
    for _ in range(num_polynomials):
        # Generate first polynomial of degree i

        i = random.randint(1, C-1)

        operation = random.choice(["add", "multiply"])

        first_poly = generate_random_polynomials_recursive(n, d, i, mod)

        second_poly = generate_random_polynomials_recursive(n, d, C-i, mod)

        if operation == 'add':
            poly_dict=add_polynomials(first_poly, second_poly, d, mod)
        else:
            poly_dict = multiply_polynomials(first_poly, second_poly, d, mod)
        
        # Convert to vector
        vector = [0] * len(all_monomials)
        for monomial, coef in poly_dict.items():
            if monomial in monomial_to_index:
                vector[monomial_to_index[monomial]] = coef
        
        all_polynomials.append(vector)
    
    return index_to_monomial, monomial_to_index, all_polynomials

def generate_random_polynomials_recursive(n, d, C, mod):
    """
    Generate random polynomials by multiplying polynomials of specific degrees.
    
    Parameters:
    n - number of variables
    d - maximum degree
    C - complexity parameter
    num_polynomials - number of polynomials to generate
    
    Returns:
    index_to_monomial - dictionary mapping indices to monomials
    all_polynomials - list of polynomial vectors
    """
    
    
    # Generate first polynomial of degree i

    if C ==1:
        var_idx = random.randint(0, n)
        exponents = [0] * n
        if var_idx != n:
            exponents[var_idx] = 1
        poly = {tuple(exponents): 1}

        return poly
    else:
        i = random.randint(1, C-1)

        operation = random.choice(["add", "multiply"])

        first_poly = generate_random_polynomials_recursive(n, d, i, mod)

        second_poly = generate_random_polynomials_recursive(n, d, C-i, mod)

        if operation == 'add':
            poly_dict = add_polynomials(first_poly, second_poly, d, mod)
        else:
            poly_dict = multiply_polynomials(first_poly, second_poly, d, mod)
        

    return poly_dict

def add_polynomials(poly1, poly2, d, mod):
    """Add two polynomials, restricting to terms of degree <= d"""
    result = poly1.copy()
    
    for monomial, coef in poly2.items():
        if sum(monomial) > d:
            continue  # Skip terms with degree > n
        
        if monomial in result:
            result[monomial] = (result[monomial] + coef) % mod #mod2
            if result[monomial] == 0:
                del result[monomial]
        else:
            result[monomial] = coef
    
    return result

def multiply_polynomials(poly1, poly2, d, mod):
    """Multiply two polynomials, restricting to terms of degree <= n"""
    result = {}
    
    for mon1, coef1 in poly1.items():
        for mon2, coef2 in poly2.items():
            # Add the exponents
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            
            # Skip terms with degree > d
            if sum(new_mon) > d:
                continue
            
            # Multiply the coefficients mod2
            new_coef = (coef1 * coef2) % 2
            
            # Add to the result
            if new_mon in result:
                result[new_mon] = (result[new_mon] + new_coef) % mod
                if result[new_mon] == 0:
                    del result[new_mon]
            else:
                result[new_mon] = new_coef
    
    return result
