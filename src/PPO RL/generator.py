import random
import sympy

# --- SymPy Helpers ---

_symbols = None

def get_symbols(n: int):
    """Get a cached list of SymPy symbols x0, x1, ..."""
    global _symbols
    if _symbols is None or len(_symbols) < n:
        _symbols = sympy.symbols(f"x0:{n}")
    return _symbols

def _canonical_key(expr: sympy.Expr, symbols) -> str:
    """
    Creates a canonical string representation for a SymPy polynomial expression.
    This key is used for deduplication.
    """
    e = sympy.expand(expr)
    p = sympy.Poly(e, *symbols, domain='QQ')
    return sympy.srepr(p.as_expr())

# --- Public API (SymPy-based) ---

def generate_random_circuit(n: int, C: int, mod: int = 2):
    """
    Generate a random arithmetic circuit using SymPy.
    Returns the actions and the list of generated SymPy expressions.
    """
    symbols = get_symbols(n)
    
    actions: list[tuple[str, int | None, int | None]] = []
    sympy_polynomials: list[sympy.Expr] = []
    
    # Track canonical representations to avoid duplicate sub-circuits
    seen_polynomials = set()

    # Add variable nodes
    for i in range(n):
        actions.append(("input", i, -1))
        expr = symbols[i]
        sympy_polynomials.append(expr)
        seen_polynomials.add(_canonical_key(expr, symbols))

    # # Add constant node (value 1)
    # actions.append(("constant", -1, -1))
    # expr = sympy.Integer(1)
    # sympy_polynomials.append(expr)
    # seen_polynomials.add(_canonical_key(expr, symbols))

    # Add operations
    for _ in range(C):
        if len(sympy_polynomials) == 0: continue
        num_nodes = len(sympy_polynomials)
        input1_idx = random.randint(0, num_nodes - 1)
        input2_idx = random.randint(0, num_nodes - 1)
        operation = random.choice(["add", "multiply"])

        poly1 = sympy_polynomials[input1_idx]
        poly2 = sympy_polynomials[input2_idx]

        if operation == "add":
            new_expr = sympy.expand(poly1 + poly2)
        else:  # multiply
            new_expr = sympy.expand(poly1 * poly2)
        
        # Apply modulo to coefficients
        poly = sympy.Poly(new_expr, symbols, domain='ZZ')
        new_expr = sympy.Poly({m: c % mod for m, c in poly.terms()}, symbols, domain='ZZ').as_expr()

        # Deduplication check
        key = _canonical_key(new_expr, symbols)
        if key in seen_polynomials:
            continue # Skip adding this duplicate operation

        actions.append((operation, input1_idx, input2_idx))
        sympy_polynomials.append(new_expr)
        seen_polynomials.add(key)

    # Trim unused operations
    final_actions, final_sympy_polynomials, _ = trim_circuit(actions, sympy_polynomials)

    return final_actions, final_sympy_polynomials


def trim_circuit(actions, polynomials):
    """
    Trim unused operations from a circuit. This function works on the action list
    and the list of polynomials (either vector or sympy format).
    """
    if not actions:
        return [], [], {}

    used = set()
    num_inputs_constants = 0
    for i, (op, _, _) in enumerate(actions):
        if op in ("input", "constant"):
            # These are always considered used initially
            num_inputs_constants += 1

    if len(actions) <= num_inputs_constants:
        return actions, polynomials, {i: i for i in range(len(actions))}

    # Start traversal from the final output node
    stack = [len(actions) - 1]
    
    while stack:
        idx = stack.pop()
        if idx in used:
            continue
        
        used.add(idx)
        
        op, in1, in2 = actions[idx]
        if op in ("add", "multiply"):
            if in1 is not None:
                stack.append(in1)
            if in2 is not None:
                stack.append(in2)

    used_sorted = sorted(list(used))
    remap = {old: new for new, old in enumerate(used_sorted)}

    new_actions = []
    new_polynomials = []
    for old_idx in used_sorted:
        op, in1, in2 = actions[old_idx]
        if op in ("add", "multiply"):
            if in1 in remap and in2 in remap:
                new_actions.append((op, remap[in1], remap[in2]))
                new_polynomials.append(polynomials[old_idx])
        else: # input or constant
            new_actions.append((op, old_idx if op == 'input' else -1, -1))
            new_polynomials.append(polynomials[old_idx])

    return new_actions, new_polynomials, remap


def generate_random_polynomials(n, C, num_polynomials=10000, mod=5):
    """
    Generate a dataset of random polynomials and their corresponding circuits.
    """
    all_polynomials: list[sympy.Expr] = []
    all_circuits: list[list[tuple[str, int | None, int | None]]] = []
    
    final_poly_keys = set()
    symbols = get_symbols(n)

    attempts = 0
    max_attempts = num_polynomials * 5 # Stop if it's too hard to find new polys

    while len(all_polynomials) < num_polynomials and attempts < max_attempts:
        attempts += 1
        circuit, polys = generate_random_circuit(n, C, mod)
        
        if not polys:
            continue

        final_poly = polys[-1]
        
        key = _canonical_key(final_poly, symbols)
        if key not in final_poly_keys:
            all_circuits.append(circuit)
            all_polynomials.append(final_poly)
            final_poly_keys.add(key)

            if len(all_polynomials) % 1000 == 0:
                print(f"Generated {len(all_polynomials)}/{num_polynomials} unique polynomials...")

    if attempts >= max_attempts:
        print(f"Warning: Stopped after {max_attempts} attempts. Generated {len(all_polynomials)} polynomials.")

    return all_polynomials, all_circuits, attempts
