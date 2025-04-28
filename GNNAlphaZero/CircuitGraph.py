class CircuitGraph:
    def __init__(self, all_monomials):
        """
        Initialize a CircuitGraph.

        Args:
            all_monomials (List[Tuple[int]]): The shared monomial basis for all polynomial vectors.
        """
        self.nodes = []  # list of polynomial vectors (each is List[int])
        self.edges = []  # list of (parent1_idx, parent2_idx, operation) tuples
        self.all_monomials = all_monomials  # basis used for interpreting vectors

    def add_node(self, vector):
        """
        Add a new node to the graph.

        Args:
            vector (List[int]): Polynomial vector over the monomial basis.

        Returns:
            int: Index of the newly added node.
        """
        self.nodes.append(vector)
        return len(self.nodes) - 1

    def add_edge(self, parent1_idx, parent2_idx, operation, result_vector):
        """
        Add a new node resulting from applying an operation between two parent nodes,
        and record the operation edge.

        Args:
            parent1_idx (int): Index of the first parent node.
            parent2_idx (int): Index of the second parent node.
            operation (str): Either "add" or "multiply".
            result_vector (List[int]): Resulting polynomial vector after operation.

        Returns:
            int: Index of the newly added result node.
        """
        assert operation in ["add", "multiply"], "Operation must be 'add' or 'multiply'."
        result_idx = self.add_node(result_vector)
        self.edges.append((parent1_idx, parent2_idx, operation))
        return result_idx

    def num_nodes(self):
        return len(self.nodes)

    def num_edges(self):
        return len(self.edges)

    def __repr__(self):
        return f"CircuitGraph(num_nodes={self.num_nodes()}, num_edges={self.num_edges()})"