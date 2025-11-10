# Migration to Pure SymPy-Based Polynomial Generation

This document outlines the refactoring of `generator.py` and its consumers from a hybrid vector/symbolic system to a pure `sympy`-based API. This change introduces **breaking changes** to the public API in favor of a more robust, correct, and maintainable codebase.

## 1. API Breaking Changes

The public-facing API of `generator.py` has been modified. Functions no longer return vector-based representations of polynomials.

### `generate_random_polynomials`

-   **Old Signature**: `generate_random_polynomials(n, d, C, num_polynomials, mod)`
-   **Old Return**: `(index_to_monomial, monomial_to_index, all_polynomials_vectors, all_circuits)`
-   **New Signature**: `generate_random_polynomials(n, C, num_polynomials, mod)` (degree `d` is no longer needed)
-   **New Return**: `(all_polynomials_expressions, all_circuits)`

### `generate_random_circuit`

-   **Old Signature**: `generate_random_circuit(n, d, C, mod)`
-   **Old Return**: `(actions, polynomials_vectors, index_to_monomial, monomial_to_index)`
-   **New Signature**: `generate_random_circuit(n, C, mod)`
-   **New Return**: `(actions, polynomials_expressions)`

## 2. Canonical Key and Deduplication

To reliably identify and deduplicate algebraically equivalent polynomials, a canonical key is generated.

### Canonical Key Method

The internal `_canonical_key` function implements the following exact steps to produce a reproducible key:
1.  **`sympy.expand()`**: The expression is fully expanded into a standard sum-of-products form.
2.  **`sympy.Poly()`**: The expanded expression is converted to a `sympy.Poly` object with `domain='QQ'` (rational numbers) to normalize its internal structure and order terms deterministically.
3.  **`.as_expr()`**: The `Poly` object is converted back to a canonical `sympy.Expr`.
4.  **`sympy.srepr()`**: A stable string representation of the canonical expression is generated, which is used as the deduplication key.

### Deduplication Strategy

Deduplication happens **during generation**, not as a post-hoc filter.
-   In `generate_random_circuit`, a `seen_polynomials` set tracks the canonical keys of all sub-expressions. If a newly generated expression is already in the set, the operation is skipped, preventing redundant work.
-   In `generate_random_polynomials`, a `final_poly_keys` set ensures that only unique final polynomials are added to the dataset, building a unique list from the start.

## 3. Symbols Definition

Polynomial variables are consistently defined as `x0, x1, ...`. This is managed by the `get_symbols(n)` helper function, which creates and caches `sympy.symbols(f'x0:{n}')`. This function is the single source of truth for symbols, ensuring consistency across the generator and consumers.

## 4. Performance Guardrails

-   **Efficient Operations**: Performance-critical loops avoid `sympy.simplify()`, which can be slow. Instead, the faster `sympy.expand()` is used for normalization.
-   **Performance Test**: The test suite in `tests/test_generator.py` includes a `test_performance_generation` case. This provides a basic benchmark to guard against significant performance regressions in the polynomial generation process.

## 5. API and Consumer Usage Examples

### New `generate_random_polynomials` Return Type

```python
# Example of the new API usage
from transformer.generator import generate_random_polynomials

n, C, num_p, mod = 3, 5, 100, 7

# The function now returns SymPy expressions directly
all_polynomials_expr, all_circuits = generate_random_polynomials(
    n, C, num_polynomials=num_p, mod=mod
)

# Example output:
# all_polynomials_expr[0] might be: x0*x1 + 2*x2 + 1
print(f"Generated {len(all_polynomials_expr)} unique polynomials.")
print(f"First polynomial: {all_polynomials_expr[0]}")
```

### Consumer Update in `fourthGen.py`

The `CircuitDataset` now consumes `sympy.Expr` objects and converts them to tensors using a new `sympy_to_tensor` utility.

```python
# In fourthGen.py

def sympy_to_tensor(expr: sympy.Expr, n_vars: int, max_degree: int) -> torch.Tensor:
    """Converts a SymPy expression into a flattened tensor."""
    symbols = get_symbols(n_vars)
    poly = sympy.Poly(expr, symbols)
    tensor_shape = tuple([max_degree + 1] * n_vars)
    tensor = torch.zeros(tensor_shape, dtype=torch.float)
    for exponents, coeff in poly.terms():
        if all(e <= max_degree for e in exponents):
            tensor[exponents] = float(coeff)
    return tensor.flatten()

class CircuitDataset(Dataset):
    def __init__(self, config, size, description):
        # ...
        # 1. Get SymPy expressions from the generator
        all_polynomials, all_circuits = generate_random_polynomials(
            n, C, num_polynomials=num_circuits, mod=self.config.mod
        )

        for i, target_poly_expr in enumerate(all_polynomials):
            # 2. Convert expression to tensor for the model
            target_poly_tensor = sympy_to_tensor(target_poly_expr, n, self.config.max_degree)
            # ... rest of dataset processing
```

## 6. Model and Configuration Migration

**CRITICAL**: This refactoring introduces changes that make previous model checkpoints and configurations obsolete.

-   **Model Input Size**: The input shape for the `polynomial_embedding` layer in the `CircuitBuilder` model has changed. It is now dynamically calculated as `(max_degree + 1) ** n_variables`.
-   **Action Required**:
    1.  **Delete Old Checkpoints**: Pre-trained models (e.g., `best_supervised_model_*.pt`, `ppo_model_*.pt`) are incompatible and must be deleted to allow for retraining.
    2.  **Configuration Update**: The model configuration no longer depends on a pre-calculated `max_vector_size`. The size is determined at runtime in `fourthGen.py`. Ensure your scripts reflect this new dynamic sizing.
```
