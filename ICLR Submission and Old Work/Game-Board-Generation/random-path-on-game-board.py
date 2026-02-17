import random
import sympy

# --- SymPy Helpers ---

random.seed()
def generate_random_x_circuit(C: int):
    """
    Generate a random arithmetic circuit using SymPy.
    Returns the actions and the list of generated SymPy expressions.
    """
    x = sympy.symbols('x')
    
    actions: list[tuple[str, int | None, int | None]] = []
    states: list[sympy.Expr] = [x] #store unique states

<<<<<<< HEAD
=======


>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
    # Add operations
    for i in range(C):
        if len(states) == 0: continue
        num_nodes = len(states)
        input1_idx = random.randint(0, num_nodes - 1)
        operation = random.choice(["add", "multiply"])

        poly1 = states[input1_idx]
        poly2 = states[-1]

        if operation == "add":
            new_expr = sympy.expand(poly1 + poly2)
        else:  # multiply
            new_expr = sympy.expand(poly1 * poly2)
        

        # Deduplication check
        

        actions.append((operation, input1_idx, i))
        states.append(new_expr)



    return actions, states

