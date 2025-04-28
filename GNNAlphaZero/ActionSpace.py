import torch

def enumerate_all_actions(graph):
    """
    Enumerates all valid actions from the current graph.

    Args:
        graph (CircuitGraph): The current circuit graph.

    Returns:
        List[Tuple[int, int, int]]: (parent1_idx, parent2_idx, op_type)
                                    op_type: 0 = add, 1 = multiply
    """
    actions = []
    num_nodes = len(graph.nodes)
    for i in range(num_nodes):
        for j in range(num_nodes):
            for op in [0, 1]:  # 0: add, 1: multiply
                actions.append((i, j, op))
    return actions


def score_actions(actions, policy_logits):
    """
    Scores actions based on node policy logits.

    Args:
        actions (List[Tuple[int, int, int]]): List of (parent1_idx, parent2_idx, op_type).
        policy_logits (torch.Tensor): (num_nodes,) node logits from the GNN.

    Returns:
        torch.Tensor: (num_actions,) scores for each action.
    """
    scores = []
    for (i, j, op) in actions:
        op_bonus = 0.0 if op == 0 else 0.1  # slight bias to multiplication if you want
        score = policy_logits[i] + policy_logits[j] + op_bonus
        scores.append(score)
    return torch.stack(scores)