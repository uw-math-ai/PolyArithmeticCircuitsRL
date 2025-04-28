import random
from itertools import combinations_with_replacement
import sympy as sp

# ——— Modification: cache SymPy symbols to avoid recreating them on every call ———
SYMBOL_CACHE = {}


def sympy_to_vector(expr, monomial_list, variable_names=None):
    """
    Converts a SymPy polynomial expression into a binary vector representation.

    Args:
        expr (sympy.Expr): The polynomial expression to convert.
        monomial_list (List[Tuple[int]]): The exponent tuples defining the monomial basis.
        variable_names (List[str], optional): List of variable names. If None, uses ['x0', 'x1', ..., 'xm'].

    Returns:
        List[int]: Binary vector (0/1) over the monomial basis, reduced modulo 2.
    """
    if expr == 0:
        return [0] * len(monomial_list)

    # Infer number of variables from monomial list
    m = len(monomial_list[0])
    if variable_names is None:
        variable_names = [f'x{i}' for i in range(m)]
    vars = sp.symbols(variable_names)

    # Expand and collect coefficients of all monomials
    expr = sp.expand(expr)
    terms = expr.as_coefficients_dict()

    vector = [0] * len(monomial_list)
    monomial_index = {mon: i for i, mon in enumerate(monomial_list)}

    for term, coeff in terms.items():
        if coeff % 2 == 0:
            continue  # ignore zero or even coefficients in ℱ₂

        # Build exponent tuple
        if term == 1:
            exponents = (0,) * m
        else:
            exponents = [0] * m
            if isinstance(term, sp.Symbol):
                idx = variable_names.index(str(term))
                exponents[idx] = 1
            else:
                for factor in term.as_ordered_factors():
                    if isinstance(factor, sp.Symbol):
                        idx = variable_names.index(str(factor))
                        exponents[idx] += 1
                    elif isinstance(factor, sp.Pow):
                        base, exp = factor.args
                        idx = variable_names.index(str(base))
                        exponents[idx] += int(exp)
        exponents = tuple(exponents)

        if exponents in monomial_index:
            idx = monomial_index[exponents]
            vector[idx] = 1  # in ℱ₂

    return vector


def vector_to_sympy(vector, monomial_list, variable_names=None):
    """
    Input:
        vector (List[int]): Binary vector of coefficients (length = len(monomial_list))
        monomial_list (List[Tuple[int]]): List of exponent tuples for each monomial
        variable_names (List[str], optional): Variable names to use (defaults to ['x0', 'x1', ..., 'xm'])

    Output:
        The polynomail in SymPy expression
    """
    m = len(monomial_list[0])  # number of variables
    if variable_names is None:
        variable_names = [f'x{i}' for i in range(m)]

    # ——— Modification: look up or create symbols only once per variable_names tuple ———
    key = tuple(variable_names)
    vars = SYMBOL_CACHE.get(key)
    if vars is None:
        vars = sp.symbols(variable_names)
        SYMBOL_CACHE[key] = vars

    expr = 0
    for coef, exponents in zip(vector, monomial_list):
        if coef == 0:
            continue
        term = 1
        for var, power in zip(vars, exponents):
            term *= var ** power
        expr += term
    # ——— Modification: drop the expensive simplify() call ———
    return expr


def generate_random_polynomials(n, m, C, num_polynomials=10000):

    all_monomials = []
    for total_degree in range(n + 1):
        for combo in combinations_with_replacement(range(m), total_degree):
            exponents = [0] * m
            for var_idx in combo:
                exponents[var_idx] += 1
            
            all_monomials.append(tuple(exponents))
    
    all_monomials.sort()

    # indexed list
    monomial_to_index = {monomial: idx for idx, monomial in enumerate(all_monomials)}

    all_polynomials = []
    
    for _ in range(num_polynomials):
        poly = generate_random_polynomial(n, m, C)
        
        # Convert to vector
        vector = [0] * len(all_monomials)
        for monomial, coef in poly.items():
            if monomial in monomial_to_index:
                vector[monomial_to_index[monomial]] = coef
        
        all_polynomials.append(vector)
    
    return all_polynomials, all_monomials

def generate_random_polynomial(n, m, C):

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
    
    # print(f"poly:  {poly}")
    return poly

def add_polynomials(poly1, poly2, n):
    result = poly1.copy()
    
    for monomial, coef in poly2.items():
        if sum(monomial) > n:
            continue  # Skip terms with degree > n
        
        if monomial in result:
            result[monomial] = (result[monomial] + coef) % 2
            if result[monomial] == 0:
                del result[monomial]
        else:
            result[monomial] = coef
    
    return result

def multiply_polynomials(poly1, poly2, n):
    result = {}
    
    for mon1, coef1 in poly1.items():
        for mon2, coef2 in poly2.items():
            # Add the exponents
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            
            # Skip terms with degree > n
            if sum(new_mon) > n:
                continue
            
            # Multiply the coefficients (both are 1, so result is 1)
            new_coef = coef1 * coef2  # This will always be 1
            
            # Add to the result
            if new_mon in result:
                result[new_mon] = (result[new_mon] + new_coef) % 2
                if result[new_mon] == 0:
                    del result[new_mon]
            else:
                result[new_mon] = new_coef
    
    return result

# polyns, all_monomials = generate_random_polynomials(5,6,4)
#
# for polyn in polyns:
#     sym_expr = vector_to_sympy(polyn, all_monomials)
#     print(sym_expr)
