import math
import random
import torch
from ActionSpace import enumerate_all_actions, score_actions
from GraphConverter import circuitgraph_to_graphdata
from copy import deepcopy
from VectorOps import *

class MCTSAgent:
    def __init__(self, model, simulations_per_move=30, c_puct=1.5):
        self.model = model
        self.simulations_per_move = simulations_per_move
        self.c_puct = c_puct

    def select_action(self, root_graph):
        """
        Perform MCTS starting from root_graph.
        Returns the selected action.
        """
        # MCTS statistics
        N = {}  # visit count
        W = {}  # total value
        Q = {}  # mean value
        P = {}  # prior probability

        actions = enumerate_all_actions(root_graph)

        # Evaluate root node with GNN
        graph_data = circuitgraph_to_graphdata(root_graph)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
        policy_logits, root_value = self.model(graph_data)
        policy_logits = policy_logits.squeeze()

        prior_scores = score_actions(actions, policy_logits)
        prior_scores = torch.softmax(prior_scores, dim=0)

        for idx, action in enumerate(actions):
            P[action] = prior_scores[idx].item()
            N[action] = 0
            W[action] = 0.0
            Q[action] = 0.0

        # MCTS simulations
        for _ in range(self.simulations_per_move):
            self.simulate(deepcopy(root_graph), N, W, Q, P)

        # After simulations: pick action with highest visit count
        best_action = max(actions, key=lambda a: N[a])

        return best_action

    def simulate(self, graph, N, W, Q, P):
        path = []
        actions = enumerate_all_actions(graph)

        # --- Patch Start: initialize missing actions ---
        for a in actions:
            if a not in N:
                N[a] = 0
                W[a] = 0.0
                Q[a] = 0.0
                P[a] = 1e-8  # very small prior for unseen actions
        # --- Patch End ---

        # Selection
        while True:
            total_N = sum(N[a] for a in actions) + 1e-8
            ucb_scores = []
            for a in actions:
                ucb = Q[a] + self.c_puct * P[a] * math.sqrt(total_N) / (1 + N[a])
                ucb_scores.append(ucb)
            best_idx = int(torch.argmax(torch.tensor(ucb_scores)))
            best_action = actions[best_idx]

            path.append(best_action)

            # Apply action
            parent1, parent2, op = best_action
            if op == 0:
                new_vec = add_vectors(graph.nodes[parent1], graph.nodes[parent2])
            else:
                new_vec = multiply_vectors(graph.nodes[parent1], graph.nodes[parent2], graph.all_monomials)
            graph.add_edge(parent1, parent2, "add" if op == 0 else "multiply", new_vec)

            # New actions after graph changes
            actions = enumerate_all_actions(graph)

            # --- Patch Start: initialize missing actions after step ---
            for a in actions:
                if a not in N:
                    N[a] = 0
                    W[a] = 0.0
                    Q[a] = 0.0
                    P[a] = 1e-8
            # --- Patch End ---

            if len(actions) == 0 or len(graph.nodes) > 30:
                break

        # Evaluate leaf
        graph_data = circuitgraph_to_graphdata(graph)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
        _, value = self.model(graph_data)
        value = value.item()

        # Backup
        for a in path:
            N[a] += 1
            W[a] += value
            Q[a] = W[a] / N[a]
