# Documentation for `generator.py`

This document provides a detailed explanation of the `transformer/generator.py` module, which is responsible for generating random arithmetic circuits and their corresponding polynomial expressions using the `SymPy` library.

## 1. Overview

The primary purpose of `generator.py` is to create a dataset of complex polynomials and the step-by-step arithmetic circuits that produce them. This dataset is fundamental for training machine learning models to understand the relationship between a computational graph (the circuit) and its mathematical result (the polynomial).

The generator ensures that the generated polynomials are unique and that the circuits are efficient (i.e., they don't contain unnecessary operations). The entire public API is based on `SymPy`, providing a robust and mathematically precise way to handle the polynomials.

---

## 2. Core Components & Logic

### 2.1. Symbol Management (`get_symbols`)

- **Purpose**: To provide a consistent, cached list of `SymPy` symbols (e.g., `x0`, `x1`, `x2`, ...).
- **Implementation**: A global cache `_symbols` stores the list of variables. The `get_symbols(n)` function generates `n` symbols if they haven't been created yet or if more are requested than are currently cached. This avoids the overhead of recreating symbols on every call.

### 2.2. Canonical Representation for Deduplication (`_canonical_key`)

This is the most critical component for ensuring the uniqueness of generated polynomials.

- **Purpose**: To create a single, consistent string representation for any given `SymPy` polynomial expression, regardless of how it was constructed. For example, `x0 + x1` and `x1 + x0` should produce the same key.
- **Implementation**: The function follows a multi-step process to normalize an expression:
    1.  `sympy.expand(expr)`: It first fully expands the expression. This resolves all parenthesis and combines terms, e.g., `(x0+1)*x1` becomes `x0*x1 + x1`.
    2.  `sympy.Poly(..., domain='QQ')`: It then converts the expanded expression into a `SymPy` `Poly` object. This object represents the polynomial in a structured way, with its terms and coefficients explicitly defined. Using the rational numbers domain (`'QQ'`) ensures that coefficients are handled precisely.
    3.  `p.as_expr()`: It converts the `Poly` object back into a standard `Expr` object. This step sorts the terms into a canonical order (e.g., by degree and then lexicographically).
    4.  `sympy.srepr()`: Finally, it generates the "standard representation" string of the expression. This is an unambiguous string that uniquely identifies the structure of the expression, making it perfect for use as a dictionary key or for adding to a set.

### 2.3. Random Circuit Generation (`generate_random_circuit`)

- **Purpose**: To build a single, random arithmetic circuit of a given complexity.
- **Implementation**:
    1.  **Initialization**:
        - It starts by creating "input" nodes for each variable (e.g., `x0`, `x1`).
        - It adds a "constant" node, which is always `sympy.Integer(1)`.
        - The `sympy.Expr` for each of these initial nodes is stored, and their canonical keys are added to a `seen_polynomials` set to begin the deduplication process.
    2.  **Iterative Operation Generation**:
        - It loops `C` times (where `C` is the desired circuit complexity).
        - In each iteration, it randomly selects an operation (`"add"` or `"multiply"`) and two existing nodes from the list of polynomials generated so far.
        - It applies the operation to the two corresponding `SymPy` expressions.
    3.  **Modulo Arithmetic**:
        - After creating the new expression (e.g., `poly1 + poly2`), it is expanded.
        - The result is converted to a `sympy.Poly` object. This allows direct access to the coefficients of each term.
        - A new `Poly` object is created where each coefficient `c` is replaced by `c % mod`.
        - This is then converted back to a standard `sympy.Expr`. This ensures all arithmetic is performed within the specified finite field.
    4.  **Deduplication Check**:
        - It calculates the `_canonical_key` for the newly generated expression.
        - If the key is already in the `seen_polynomials` set, the operation is discarded, and the loop continues. This prevents adding redundant sub-circuits that produce an already-existing polynomial.
        - If the key is new, the action (e.g., `("add", 5, 2)`) and the new `sympy.Expr` are stored, and the key is added to the `seen_polynomials` set.
    5.  **Final Trimming**: After the loop, the circuit is passed to `trim_circuit` to remove any operations that don't contribute to the final output polynomial.

### 2.4. Circuit Trimming (`trim_circuit`)

- **Purpose**: To remove "dead code" from the generated circuit. A random generator might create intermediate polynomials that are never used in the final result. This function cleans up the circuit to make it minimal.
- **Implementation**:
    1.  **Identify Used Nodes**: It performs a traversal (using a stack) starting from the *last* node in the circuit (the final output).
    2.  **Back-Propagation**: It follows the input indices (`in1`, `in2`) of each operation recursively, adding every visited node index to a `used` set.
    3.  **Rebuild the Circuit**:
        - It creates a new, sorted list of the used node indices.
        - It creates a `remap` dictionary to map the old indices to new, contiguous indices (e.g., `{0:0, 2:1, 5:2}`).
        - It iterates through the sorted list of used old indices, creating a `new_actions` list and a `new_polynomials` list. The input indices for `add` and `multiply` operations are updated using the `remap` dictionary.

### 2.5. Main Data Generation (`generate_random_polynomials`)

- **Purpose**: This is the main public function used to generate a large dataset of unique polynomials and their corresponding circuits.
- **Implementation**:
    1.  **Looping and Generation**: It repeatedly calls `generate_random_circuit` to create new circuits and polynomials.
    2.  **Uniqueness Check**: It only considers the *final* polynomial from each generated circuit. It calculates the `_canonical_key` for this final polynomial.
    3.  **Storing Unique Results**: If the key for the final polynomial has not been seen before, the circuit and the polynomial are added to the final dataset lists (`all_circuits`, `all_polynomials`), and the key is stored in `final_poly_keys`.
    4.  **Termination**: The loop continues until the desired `num_polynomials` has been generated or a maximum number of attempts is reached (to prevent infinite loops if it becomes too difficult to find new unique polynomials).

---

## 3. API and Usage

### Public Functions

- `generate_random_polynomials(n, C, num_polynomials=10000, mod=5)`
  - **`n`**: The number of input variables (e.g., `x0, x1, ...`).
  - **`C`**: The complexity (number of operations) for each circuit.
  - **`num_polynomials`**: The target number of unique polynomials to generate for the dataset.
  - **`mod`**: The modulus for all arithmetic operations.
  - **Returns**: A tuple `(all_polynomials, all_circuits, attempts)`, where:
    - `all_polynomials`: A list of unique `sympy.Expr` objects.
    - `all_circuits`: A list of the corresponding minimal circuits (action lists).
    - `attempts`: The total number of circuits that were generated to produce the final unique set.

- `generate_random_circuit(n: int, C: int, mod: int = 2)`
  - **Parameters**: Same as above.
  - **Returns**: A tuple `(final_actions, final_sympy_polynomials)`, where:
    - `final_actions`: The minimal list of actions for the generated circuit.
    - `final_sympy_polynomials`: The list of all `sympy.Expr` objects created at each step of the minimal circuit. The last element is the final output.

### Example Usage

Here is a simple example of how to use the generator:

```python
from transformer.generator import generate_random_polynomials

# Parameters
NUM_VARIABLES = 3
CIRCUIT_COMPLEXITY = 5
DATASET_SIZE = 10
MODULO = 2

print("Generating a dataset of polynomials...")

# Generate the data
polys, circuits, attempts = generate_random_polynomials(
    n=NUM_VARIABLES,
    C=CIRCUIT_COMPLEXITY,
    num_polynomials=DATASET_SIZE,
    mod=MODULO
)

print(f"Successfully generated {len(polys)} unique polynomials from {attempts} attempts.")

# Print the first generated polynomial and its circuit
if polys:
    print("\n--- Example Entry ---")
    print(f"Polynomial: {polys[0]}")
    print("Circuit Actions:")
    for i, action in enumerate(circuits[0]):
        print(f"  Step {i}: {action}")

```