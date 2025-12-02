import random
import sympy as sp


def random_polynomial(n_variables: int, max_degree: int, max_terms: int = 5, coeff_range=(-2, 2)) -> sp.Expr:
    """
    Generate a random polynomial with bounded degree and integer coefficients.
    No constant term - only terms with variables.
    """
    symbols = sp.symbols(f"x0:{n_variables}")
    terms = []
    for _ in range(random.randint(1, max_terms)):
        exponents = [random.randint(0, max_degree) for _ in range(n_variables)]
        # Skip if all exponents are 0 (would be a constant)
        if all(e == 0 for e in exponents):
            continue
        coeff = random.randint(coeff_range[0], coeff_range[1])
        if coeff == 0:
            continue
        monom = coeff
        for sym, exp in zip(symbols, exponents):
            monom *= sym**exp
        terms.append(monom)
    
    # Ensure we have at least one term
    if not terms:
        # Create a simple term like x0 or x1
        var_idx = random.randint(0, n_variables - 1)
        coeff = random.choice([c for c in range(coeff_range[0], coeff_range[1] + 1) if c != 0])
        terms.append(coeff * symbols[var_idx])
    
    poly = sum(terms)
    return sp.expand(poly)
