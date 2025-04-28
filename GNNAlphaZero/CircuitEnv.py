from GNNAlphaZero.CircuitGraph import CircuitGraph
from VectorOps import add_vectors, multiply_vectors

class CircuitEnv:
    def __init__(self, all_monomials, target_vector, max_steps=10):
        self.graph = CircuitGraph(all_monomials)
        self.target_vector = target_vector
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, init_vectors):
        """
        Initialize the environment with given initial polynomials.
        """
        self.graph = CircuitGraph(self.graph.all_monomials)
        for vec in init_vectors:
            self.graph.add_node(vec)
        self.steps = 0
        return self.graph

    def step(self, action):
        parent1, parent2, op = action
        if op == 0:
            new_vec = add_vectors(self.graph.nodes[parent1], self.graph.nodes[parent2])
        else:
            new_vec = multiply_vectors(self.graph.nodes[parent1], self.graph.nodes[parent2], self.graph.all_monomials)

        child_idx = self.graph.add_edge(parent1, parent2, "add" if op == 0 else "multiply", new_vec)
        self.steps += 1

        # Compute reward
        reward = self.compute_reward(self.graph.nodes[child_idx])

        done = self.steps >= self.max_steps or reward >= 0.99  # allow early stopping if match close enough
        return self.graph, reward, done

    def compute_reward(self, current_vector):
        """
        Partial reward based on Hamming similarity.
        Reward is in [0, 1], 1 means perfect match.
        """
        match = sum(1 for a, b in zip(current_vector, self.target_vector) if a == b)
        total = len(current_vector)
        return match / total

    def is_target_reached(self, node_idx):
        return self.graph.nodes[node_idx] == self.target_vector
