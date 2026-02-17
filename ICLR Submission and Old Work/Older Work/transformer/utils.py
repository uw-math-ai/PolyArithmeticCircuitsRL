import sympy as sp

def encode_action(operation, node1_id, node2_id, max_nodes):
    """
    Encodes an action into a unique integer ID.
    Commutative operations (add, multiply) are handled by sorting node IDs.
    """
    if node1_id > node2_id:
        node1_id, node2_id = node2_id, node1_id

    # This formula maps a pair (i, j) with i <= j to a unique index.
    # It's the index in a flattened upper-triangular matrix.
    offset = node1_id * max_nodes - (node1_id * (node1_id - 1)) // 2
    pair_idx = offset + (node2_id - node1_id)

    op_idx = 1 if operation == "multiply" else 0
    return pair_idx * 2 + op_idx

def decode_action(action_idx, max_nodes):
    """
    Decodes an action ID back into the operation and node IDs.
    """
    op_idx = action_idx % 2
    pair_idx = action_idx // 2
    
    operation = "multiply" if op_idx == 1 else "add"

    # Inverse of the encoding formula to find node1_id
    # We are looking for node1_id such that:
    # node1_id * max_nodes - (node1_id * (node1_id - 1)) / 2 <= pair_idx
    node1_id = 0
    for i in range(max_nodes):
        offset = i * max_nodes - (i * (i - 1)) // 2
        if offset > pair_idx:
            break
        node1_id = i

    offset_for_node1 = node1_id * max_nodes - (node1_id * (node1_id - 1)) // 2
    node2_id = pair_idx - offset_for_node1 + node1_id
    
    return operation, node1_id, int(node2_id)
