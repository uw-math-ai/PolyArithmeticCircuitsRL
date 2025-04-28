def add_vectors(v1, v2):
    """
    Add two polynomial vectors over ℱ₂.

    Args:
        v1 (List[int]): First polynomial vector.
        v2 (List[int]): Second polynomial vector.

    Returns:
        List[int]: Resulting polynomial vector.
    """
    assert len(v1) == len(v2), "Vectors must have the same length."
    return [(a + b) % 2 for a, b in zip(v1, v2)]


def multiply_vectors(v1, v2, all_monomials, max_total_degree=None):
    """
    Multiply two polynomial vectors over ℱ₂.

    Args:
        v1 (List[int]): First polynomial vector.
        v2 (List[int]): Second polynomial vector.
        all_monomials (List[Tuple[int]]): The shared monomial basis.
        max_total_degree (int, optional): If given, skip terms exceeding this total degree.

    Returns:
        List[int]: Resulting polynomial vector.
    """
    # Reconstruct sparse dicts
    poly1 = {}
    poly2 = {}
    for idx, coef in enumerate(v1):
        if coef:
            poly1[all_monomials[idx]] = 1
    for idx, coef in enumerate(v2):
        if coef:
            poly2[all_monomials[idx]] = 1

    # Multiply the sparse polynomials
    result_poly = {}
    for mon1 in poly1:
        for mon2 in poly2:
            new_mon = tuple(e1 + e2 for e1, e2 in zip(mon1, mon2))
            if (max_total_degree is not None) and (sum(new_mon) > max_total_degree):
                continue
            if new_mon in result_poly:
                result_poly[new_mon] = (result_poly[new_mon] + 1) % 2
                if result_poly[new_mon] == 0:
                    del result_poly[new_mon]
            else:
                result_poly[new_mon] = 1

    # Convert back to vector
    monomial_to_index = {mon: idx for idx, mon in enumerate(all_monomials)}
    result_vector = [0] * len(all_monomials)
    for monomial in result_poly:
        if monomial in monomial_to_index:
            result_vector[monomial_to_index[monomial]] = 1

    return result_vector