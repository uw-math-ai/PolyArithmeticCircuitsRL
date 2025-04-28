from ActionSpace import enumerate_all_actions, score_actions
from GraphConverter import circuitgraph_to_graphdata
import torch

class MockAgent:
    def __init__(self, model):
        self.model = model

    def select_action(self, graph):
        """
        Selects an action based on policy logits (greedy).
        """
        graph_data = circuitgraph_to_graphdata(graph)
        graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)

        policy_logits, value = self.model(graph_data)
        policy_logits = policy_logits.squeeze()

        actions = enumerate_all_actions(graph)
        scores = score_actions(actions, policy_logits)

        # Pick best action greedily
        best_idx = torch.argmax(scores)
        best_action = actions[best_idx]

        return best_action
