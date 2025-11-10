from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):

    @abstractmethod
    def update(self, node1_id: int, node2_id: int, op_type: int):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_encoding(self) -> np.array: 
        pass


class CompactOneHotGraphEncoder(Encoder): 
    def __init__(self, N: int, P: int, D: int):
        self.N = N
        self.P = P
        self.D = D

        self.num_op_nodes = 2 * N
        self.num_edge_nodes = N * (2 * (D + P - 1) + N - 1)
        self.num_id_nodes = N

        self.size = self.num_op_nodes + self.num_edge_nodes + self.num_id_nodes
        self.reset()

    def reset(self):
        self.circuit = np.zeros(self.size, dtype=np.float32)

        self.operation_type = self.circuit[ : self.num_op_nodes]
        self.edges = self.circuit[self.num_op_nodes : self.num_op_nodes + self.num_edge_nodes]
        self.last_node = self.circuit[-self.num_id_nodes: ]

        self.last_node_loc = 0 # represents the first node in the "unknown" set - see the readme

    """
    Expects op_type to be 0 or 1 (0 corresponds to addition, 1 to multiplication)
    node{i}_id should lie in [0, D+P-1 + N-1] and correspond to the actual id of the node being used
    """
    def update(self, node1_id: int, node2_id: int, op_type: int):
        if self.last_node_loc >= self.num_id_nodes: 
            raise ValueError("Cannot update the encoder again without resetting!")
        
        I, n = self.D + self.P - 1, self.last_node_loc
        num_ids_possible = I + n
        
        if node1_id >= num_ids_possible or node2_id >= num_ids_possible: 
            raise ValueError("Cannot connect current node to node in the future!")

        # Add operation type to circuit
        self.operation_type[2 * self.last_node_loc + op_type] = 1

        # Add two edges to circuit
        base = n * (I + I + n - 1) # formula for arithmetic series * 2
        self.edges[base + node1_id] = 1
        self.edges[base + num_ids_possible + node2_id] = 1
 
        # Update last node in circuit
        if self.last_node_loc > 0: 
            self.last_node[self.last_node_loc - 1] = 0
        
        self.last_node[self.last_node_loc] = 1
        self.last_node_loc += 1

    def get_encoding(self) -> np.array:
        return self.circuit
    
    
if __name__ == "__main__":
    e = CompactOneHotGraphEncoder(N=2, P=3, D=2)

    print(e.size)

    print(e.get_encoding())
    e.update(0, 1, 1)
    print(e.get_encoding())
    e.update(0, 1, 0)
    print(e.get_encoding())
