import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from GraphConverter import circuitgraph_to_graphdata


class SelfPlayDataset(torch.utils.data.Dataset):
    def __init__(self, experiences):
        """
        Args:
            experiences (List of (graph, action, reward))
        """
        self.graphs = []
        self.actions = []
        self.rewards = []

        for graph, action, reward in experiences:
            self.graphs.append(circuitgraph_to_graphdata(graph))
            self.actions.append(action)
            self.rewards.append(reward)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]

        p1, p2, op = action  # split action here
        return graph, torch.tensor([p1, p2, op], dtype=torch.long), torch.tensor(reward, dtype=torch.float)


class Trainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train(self, experiences, batch_size=8, epochs=5):
        dataset = SelfPlayDataset(experiences)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()

        for epoch in range(epochs):
            total_policy_loss = 0
            total_value_loss = 0
            for batch in loader:
                graphs, actions, rewards = batch

                batch_graph = graphs
                batch_size = rewards.shape[0]

                policy_logits, values = self.model(batch_graph)
                policy_logits = policy_logits.squeeze()  # shape (total_nodes_in_batch,)
                batch_indices = batch_graph.batch  # tensor: for each node, which graph it belongs to

                # For each sample in batch, pick the correct node
                selected_logits = []
                for i in range(batch_size):
                    mask = (batch_indices == i)
                    logits_in_graph = policy_logits[mask]

                    # Assume action_indices[i] is 0-based index into that graph's nodes
                    action_index = actions[i][0]  # parent1
                    selected_logit = logits_in_graph[action_index]
                    selected_logits.append(selected_logit)

                selected_logits = torch.stack(selected_logits)

                # Policy loss
                target = torch.zeros_like(selected_logits, dtype=torch.long)  # dummy target = always 0
                policy_loss = self.policy_loss_fn(selected_logits.unsqueeze(1), target)

                # Value loss
                rewards = rewards.float().unsqueeze(1)
                value_loss = self.value_loss_fn(values, rewards)

                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()

            print(f"Epoch {epoch + 1}: Policy Loss {total_policy_loss:.4f}, Value Loss {total_value_loss:.4f}")

    def batch_graphs(self, graphs):
        """
        Merge list of small graphs into one batch graph.
        """
        batch = torch.utils.data.dataloader.default_collate(graphs)
        batch.batch = batch.batch.view(-1)
        return batch

    def encode_actions(self, actions, graph_batch):
        """
        actions: tensor of shape (batch_size, 3)
        """
        parent1_indices = actions[:, 0]  # take parent1 indices
        return parent1_indices


