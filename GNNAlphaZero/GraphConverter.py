import torch
from torch_geometric.data import Data

def circuitgraph_to_graphdata(circuit_graph):
    """
    Converts a CircuitGraph to a PyTorch Geometric Data object.

    Args:
        circuit_graph (CircuitGraph): The computation graph.

    Returns:
        torch_geometric.data.Data: Graph suitable for GNN processing.
    """
    x = torch.tensor(circuit_graph.nodes, dtype=torch.float)  # (num_nodes, num_features)

    edge_index = []
    edge_attr = []

    for idx, (p1_idx, p2_idx, op) in enumerate(circuit_graph.edges):
        # Determine the child node index: nodes added later after parents
        child_idx = len(circuit_graph.nodes) - len(circuit_graph.edges) + idx

        # Connect parent1 → child and parent2 → child
        edge_index.append([p1_idx, child_idx])
        edge_index.append([p2_idx, child_idx])

        # Operation type
        op_code = 0 if op == "add" else 1
        edge_attr.append([op_code])
        edge_attr.append([op_code])

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # (2, num_edges)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)  # (num_edges, 1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data