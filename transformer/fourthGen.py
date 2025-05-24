import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import add_self_loops
import random
import os
import tqdm
import math
from torch.utils.data import Dataset, DataLoader
from generator import *
from PositionalEncoding import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from generator import generate_monomials_with_additive_indices, generate_random_circuit
from utils import encode_action, vector_to_sympy
from State import *
from State import Game
from torch.distributions import Categorical
from utils import vector_to_sympy
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

class Config:
    def __init__(self):
        # --- SIMPLIFIED SETTINGS FOR DEBUGGING ---
        self.n_variables = 2         # Reduced
        self.max_complexity = 3      # Reduced
        self.hidden_dim = 128        # Reduced
        self.embedding_dim = 128       # Reduced
        self.num_gnn_layers = 3        # Reduced
        self.num_transformer_layers = 4 # Reduced (Start even lower if needed, e.g., 2)
        self.transformer_heads = 4     # Reduced
        self.transformer_dropout = 0.2 # Increased Dropout
        self.train_size = 10000        # Kept the same, but can be reduced for speed
        self.test_size = 2000
        self.epochs = 100              # Increased Supervised Epochs
        # --- Standard Settings ---
        self.learning_rate = 0.0003
        self.batch_size = 128          # Can increase batch size for stability
        self.mod = 50
        self.max_circuit_length = 100
        self.warmup_steps = 1000
        self.weight_decay = 0.01

        # --- PPO Hyperparameters (Keep but maybe run later) ---
        self.rl_learning_rate = 1e-4
        self.ppo_iterations = 1000
        self.steps_per_batch = 4096
        self.ppo_epochs = 10
        self.ppo_minibatch_size = 128
        self.ppo_clip = 0.2
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.vf_coef = 0.5
        self.ent_coef = 0.05
        self.rl_eps = 1e-8
        self.action_temperature = 1.5
config = Config()


class ArithmeticCircuitGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(ArithmeticCircuitGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(config.num_gnn_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(config.num_gnn_layers - 1)])
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0:
            return torch.zeros(x.size(0), self.convs[-1].out_channels, device=device)
        x = F.relu(self.convs[0](x, edge_index))
        for i in range(1, len(self.convs) - 1):
            identity = x
            x = self.layer_norms[i-1](x)
            x = F.relu(self.convs[i](x, edge_index))
            x = x + identity
        x = self.convs[-1](x, edge_index)
        x = self.final_norm(x)
        return x

class CircuitBuilder(nn.Module):
    def __init__(self, config, max_poly_vector_size):
        super(CircuitBuilder, self).__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.gnn = ArithmeticCircuitGNN(4, config.hidden_dim, config.embedding_dim)
        self.circuit_encoder = CircuitHistoryEncoder(config.embedding_dim)
        self.polynomial_embedding = nn.Linear(max_poly_vector_size, config.embedding_dim)
        self.positional_encoding = PositionalEncoding(config.embedding_dim, config.max_circuit_length)
        # Added config.transformer_dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.transformer_dropout, # Added dropout
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, config.num_transformer_layers)
        # Calculate max_actions based on *current* simplified config
        max_nodes = config.n_variables + config.max_complexity + 1
        max_actions = (max_nodes * (max_nodes + 1) // 2) * 2
        self.action_head = nn.Linear(config.embedding_dim, max_actions)
        self.value_head = nn.Linear(config.embedding_dim, 1)
        self.output_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))

    def forward(self, batched_graph, target_polynomials, circuit_actions, available_actions_masks=None):
        batch_size = target_polynomials.size(0)
        node_embeddings = self.gnn(batched_graph)
        graph_embeddings = global_mean_pool(node_embeddings, batched_graph.batch)
        poly_embeddings = self.polynomial_embedding(target_polynomials)
        circuit_embeddings_list = []
        max_seq_len = 0
        for i in range(batch_size):
            tokens = self.circuit_encoder.encode_circuit_actions(circuit_actions[i])
            embeddings = self.circuit_encoder(tokens)
            circuit_embeddings_list.append(embeddings)
            max_seq_len = max(max_seq_len, embeddings.size(0))
        padded_circuit_embeddings = []
        for emb in circuit_embeddings_list:
            # Handle empty embeddings case
            if emb.nelement() == 0:
                 emb = torch.zeros(0, self.embedding_dim, device=device)

            padding = torch.zeros(max_seq_len - emb.size(0), self.embedding_dim, device=device)
            padded_circuit_embeddings.append(torch.cat([emb, padding], dim=0))

        circuit_embeddings = torch.stack(padded_circuit_embeddings, dim=1)
        circuit_embeddings = self.positional_encoding(circuit_embeddings)
        memory = torch.cat([poly_embeddings.unsqueeze(0), circuit_embeddings], dim=0)
        query = self.output_token.expand(-1, batch_size, -1)
        output = self.transformer_decoder(tgt=query, memory=memory)
        output_squeezed = output.squeeze(0)
        action_logits = self.action_head(output_squeezed)
        value_pred = self.value_head(output_squeezed).squeeze(-1)
        if available_actions_masks is not None:
            # Ensure mask size matches action_logits size
            if action_logits.size(1) > available_actions_masks.size(1):
                 # Pad mask if necessary
                 padding = torch.zeros(batch_size, action_logits.size(1) - available_actions_masks.size(1),
                                       dtype=torch.bool, device=device)
                 available_actions_masks = torch.cat([available_actions_masks, padding], dim=1)
            elif action_logits.size(1) < available_actions_masks.size(1):
                 # Truncate mask if necessary
                 available_actions_masks = available_actions_masks[:, :action_logits.size(1)]

            action_logits = action_logits.masked_fill(~available_actions_masks, float('-inf'))
        return action_logits, value_pred

    def get_action_and_value(self, state, action_idx=None, temperature=1.0):
        circuit_graph, target_poly, circuit_actions, mask = state
        batched_graph = Batch.from_data_list([circuit_graph.to(device)])
        target_poly = target_poly.to(device)
        mask = mask.to(device) # Mask needs to be 2D [1, max_actions]
        if mask.dim() == 1: mask = mask.unsqueeze(0)

        action_logits, value_pred = self.forward(batched_graph, target_poly, circuit_actions, mask)

        valid_indices = torch.where(mask[0])[0]
        if len(valid_indices) == 0: return None, None, None, value_pred

        valid_logits = action_logits[0, valid_indices]
        dist = Categorical(logits=valid_logits / temperature)

        if action_idx is None:
            local_action = dist.sample()
            action = valid_indices[local_action].item()
        else:
            action = action_idx
            local_action_tensor = (valid_indices == action).nonzero(as_tuple=True)[0]
            if len(local_action_tensor) == 0: return None, None, None, value_pred
            local_action = local_action_tensor[0]

        log_prob = dist.log_prob(local_action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value_pred

# --- CircuitDataset (No changes) ---
class CircuitDataset(Dataset):
    def __init__(self, index_to_monomial, monomial_to_index, max_vector_size, config, size=10000, description="Training"):
        self.index_to_monomial = index_to_monomial
        self.monomial_to_index = monomial_to_index
        self.max_vector_size = max_vector_size
        self.config = config # Pass config here
        print(f"Generating {description} dataset ({size} examples) with N={config.n_variables}, C={config.max_complexity}...")
        self.data = self.generate_data(size)
        print(f"Finished generating {description} dataset.")

    def generate_data(self, size):
        dataset = []
        n = self.config.n_variables
        d = self.config.max_complexity*2
        num_circuits = size // self.config.max_complexity if self.config.max_complexity > 0 else size

        for _ in tqdm.tqdm(range(num_circuits), desc=f"Generating Data (N={n}, C={self.config.max_complexity})"):
            actions, polynomials, _, _ = generate_random_circuit(n, d, self.config.max_complexity, mod=self.config.mod)
            if not polynomials: continue # Skip if generation failed
            target_poly = torch.tensor(polynomials[-1], dtype=torch.float)

            for i in range(n + 1, len(actions)):
                current_actions = actions[:i]
                next_action = actions[i]
                next_op, next_node1_id, next_node2_id = next_action
                # Use current config for mask size
                max_nodes = self.config.n_variables + self.config.max_complexity + 1
                available_mask = self.get_available_actions_mask(current_actions, max_nodes)
                action_idx = encode_action(next_op, next_node1_id, next_node2_id, max_nodes)

                if action_idx >= len(available_mask): continue

                dataset.append({
                    'actions': current_actions, 'target_poly': target_poly,
                    'mask': available_mask.cpu(), 'action': action_idx,
                    'value': (len(actions) - i) / max(1, len(actions) - n -1)
                })
                if len(dataset) >= size: return dataset
        return dataset

    def get_available_actions_mask(self, actions, max_nodes):
        n_nodes = len(actions)
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2
        mask = torch.zeros(max_possible_actions, dtype=torch.bool)
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                for op in ["add", "multiply"]:
                    action_idx = encode_action(op, i, j, max_nodes)
                    if action_idx < max_possible_actions: mask[action_idx] = True
        return mask

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        circuit_graph = self.actions_to_graph(item['actions'])
        return circuit_graph, item['target_poly'], item['actions'], item['mask'], item['action'], item['value']

    def actions_to_graph(self, actions):
        n_nodes = len(actions)
        node_features, edges = [], []
        for i, (action_type, input1_idx, input2_idx) in enumerate(actions):
            if action_type == "input": type_encoding, value = [1, 0, 0], i / max(1, self.config.n_variables)
            elif action_type == "constant": type_encoding, value = [0, 1, 0], 1.0
            else:
                type_encoding, value = [0, 0, 1], 1.0 if action_type == "multiply" else 0.0
                edges.append((input1_idx, i)); edges.append((input2_idx, i))
            node_features.append(type_encoding + [value])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
        return Data(x=x, edge_index=edge_index)

# --- circuit_collate (No changes) ---
def circuit_collate(batch):
    graphs, target_polys_list, circuit_actions, masks_list, actions_list, values_list = zip(*batch)
    batched_graph = Batch.from_data_list(graphs).to(device)
    target_polys = torch.stack(target_polys_list).to(device)
    # Ensure masks have consistent size before stacking
    max_mask_len = max(m.size(0) for m in masks_list)
    padded_masks = []
    for m in masks_list:
        padding = torch.zeros(max_mask_len - m.size(0), dtype=torch.bool)
        padded_masks.append(torch.cat([m, padding]))
    masks = torch.stack(padded_masks).to(device)

    actions = torch.tensor(actions_list, device=device)
    values = torch.tensor(values_list, dtype=torch.float, device=device)
    return batched_graph, target_polys, list(circuit_actions), masks, actions, values

# --- train_supervised (Evaluates more often) ---
def train_supervised(model, train_dataset, test_dataset, config):
    data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=circuit_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader) * config.epochs, eta_min=config.learning_rate / 100)
    best_test_acc = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_action_loss, total_value_loss, action_correct, total = 0, 0, 0, 0
        for batched_graph, target_polys, circuit_actions, masks, actions, values in tqdm.tqdm(data_loader, desc=f"Epoch {epoch+1}/{config.epochs}"):
            optimizer.zero_grad()
            action_logits, value_preds = model(batched_graph, target_polys, circuit_actions, masks)
            action_loss = F.cross_entropy(action_logits, actions)
            value_loss = F.mse_loss(value_preds, values)
            loss = action_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            action_correct += (torch.argmax(action_logits, dim=1) == actions).sum().item()
            total += actions.size(0); total_action_loss += action_loss.item() * actions.size(0); total_value_loss += value_loss.item() * actions.size(0)

        train_acc = 100*action_correct/total
        print(f"Epoch {epoch+1}: LR: {optimizer.param_groups[0]['lr']:.6f}, Train Acc: {train_acc:.2f}%, Loss: {total_action_loss/total:.4f}, VLoss: {total_value_loss/total:.4f}")

        # Evaluate every 5 epochs or on the last epoch
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            test_acc, _ = evaluate_model(model, test_dataset, config)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"*** New Best Test Accuracy ({best_test_acc:.2f}%)! Model saved to {model_path} ***")
    print(f"Finished Supervised Training. Best Test Accuracy: {best_test_acc:.2f}%")
    return model

# --- PPO (calculate_gae, train_ppo - Unchanged, but may run conditionally) ---
def calculate_gae(rewards, values, dones, gamma, lambda_gae):
    advantages = []
    gae = 0.0
    values_ext = values + [values[-1] if not dones[-1] else 0.0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values_ext[i+1] * (1.0 - dones[i]) - values_ext[i]
        gae = delta + gamma * lambda_gae * (1.0 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float, device=device), \
           torch.tensor(returns, dtype=torch.float, device=device)

def train_ppo(model, dataset, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.rl_learning_rate)
    index_to_monomial = dataset.index_to_monomial
    print("\n--- Starting PPO Training ---")
    for ppo_iter in range(config.ppo_iterations):
        model.eval()
        batch_states, batch_actions, batch_old_log_probs = [], [], []
        batch_rewards, batch_dones, batch_values = [], [], []
        collected_steps, games_played, success_count = 0, 0, 0

        with torch.no_grad():
            while collected_steps < config.steps_per_batch:
                games_played += 1
                _, target_poly, _, _, _, _ = dataset[random.randint(0, len(dataset)-1)]
                sp_target_poly = vector_to_sympy(target_poly, index_to_monomial)
                game = Game(sp_target_poly, target_poly.unsqueeze(0), config).to('cpu')
                traj_states, traj_actions, traj_log_probs, traj_rewards, traj_dones, traj_values = [], [], [], [], [], []

                while not game.is_done():
                    state_tuple = game.observe()
                    action, log_prob, _, value = model.get_action_and_value(state_tuple, temperature=config.action_temperature)
                    if action is None: break
                    game.take_action(action)
                    rewards = game.compute_rewards()
                    reward = rewards[-1] if rewards else 0.0
                    traj_states.append(state_tuple); traj_actions.append(action); traj_log_probs.append(log_prob.cpu())
                    traj_values.append(value.cpu().item()); traj_rewards.append(reward); traj_dones.append(game.is_done())

                if traj_states:
                    batch_states.extend(traj_states); batch_actions.extend(traj_actions)
                    batch_old_log_probs.extend(traj_log_probs); batch_values.extend(traj_values)
                    batch_rewards.extend(traj_rewards); batch_dones.extend(traj_dones)
                    collected_steps += len(traj_states)
                    if traj_dones[-1] and traj_rewards[-1] == 100.0: success_count += 1

        if not batch_states:
             print("PPO Iter {ppo_iter+1}: No data collected, skipping update.")
             continue

        advantages, returns = calculate_gae(batch_rewards, batch_values, batch_dones, config.gamma, config.lambda_gae)
        advantages = (advantages - advantages.mean()) / (advantages.std() + config.rl_eps)
        batch_old_log_probs_t = torch.stack(batch_old_log_probs).to(device)
        batch_actions_t = torch.tensor(batch_actions, device=device, dtype=torch.long)

        model.train()
        indices = np.arange(len(batch_states))

        for epoch in range(config.ppo_epochs):
            np.random.shuffle(indices)
            total_loss, total_pi_loss, total_v_loss, total_ent_loss = 0, 0, 0, 0
            num_batches = 0
            for i in range(0, len(indices), config.ppo_minibatch_size):
                mb_indices = indices[i:i+config.ppo_minibatch_size]
                mb_states = [batch_states[idx] for idx in mb_indices]
                mb_actions = batch_actions_t[mb_indices]; mb_old_log_probs = batch_old_log_probs_t[mb_indices]
                mb_advantages = advantages[mb_indices]; mb_returns = returns[mb_indices]
                mb_graphs = [s[0] for s in mb_states]; mb_polys = torch.cat([s[1] for s in mb_states]).to(device)
                mb_circ_actions = [s[2][0] for s in mb_states]; mb_masks = torch.cat([s[3] for s in mb_states]).to(device)
                mb_batched_graph = Batch.from_data_list(mb_graphs).to(device)
                action_logits, values_pred = model(mb_batched_graph, mb_polys, mb_circ_actions, mb_masks)
                new_log_probs, entropies = [], []
                for j, logits in enumerate(action_logits):
                    mask_j = mb_masks[j]; valid_indices = torch.where(mask_j[0])[0]
                    if len(valid_indices) == 0: continue
                    valid_logits = logits[valid_indices]; dist = Categorical(logits=valid_logits)
                    action = mb_actions[j].item(); local_action_tensor = (valid_indices == action).nonzero(as_tuple=True)[0]
                    if len(local_action_tensor) == 0:
                        full_dist = Categorical(logits=logits); new_log_probs.append(full_dist.log_prob(mb_actions[j])); entropies.append(full_dist.entropy())
                    else:
                        local_action = local_action_tensor[0]; new_log_probs.append(dist.log_prob(local_action)); entropies.append(dist.entropy())
                if not new_log_probs: continue
                new_log_probs_t = torch.stack(new_log_probs); entropies_t = torch.stack(entropies)
                ratios = torch.exp(new_log_probs_t - mb_old_log_probs)
                surr1 = ratios * mb_advantages; surr2 = torch.clamp(ratios, 1 - config.ppo_clip, 1 + config.ppo_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean(); value_loss = F.mse_loss(values_pred, mb_returns)
                entropy_loss = -entropies_t.mean(); loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); optimizer.step()
                total_loss += loss.item(); total_pi_loss += policy_loss.item(); total_v_loss += value_loss.item(); total_ent_loss += entropy_loss.item(); num_batches += 1

        success_rate = 100 * success_count / games_played if games_played > 0 else 0
        avg_ent = total_ent_loss / num_batches / (-config.ent_coef) if num_batches > 0 and config.ent_coef != 0 else 0
        print(f"PPO Iter {ppo_iter+1}: SR: {success_rate:.2f}%, L: {total_loss/num_batches:.3f} (Pi: {total_pi_loss/num_batches:.3f}, V: {total_v_loss/num_batches:.3f}, E: {avg_ent:.4f})")
        if (ppo_iter + 1) % 50 == 0: torch.save(model.state_dict(), f"ppo_model_n{config.n_variables}_C{config.max_complexity}.pt")

# --- evaluate_model (No changes) ---
def evaluate_model(model, test_dataset, config, num_tests=500):
    model.eval()
    print(f"\n--- Evaluating Supervised Performance on {min(num_tests, len(test_dataset))} Test Examples ---")
    test_indices = random.sample(range(len(test_dataset)), min(num_tests, len(test_dataset)))
    action_correct = 0
    value_mse = 0
    with torch.no_grad():
        for idx in test_indices:
            circuit_graph, target_poly, circuit_actions, mask, action, value = test_dataset[idx]
            batched_graph = Batch.from_data_list([circuit_graph]).to(device)
            target_poly = target_poly.unsqueeze(0).to(device)
            circuit_actions_batch = [circuit_actions]
            mask = mask.unsqueeze(0).to(device)
            value_tensor = torch.tensor([value], dtype=torch.float, device=device)
            action_logits, value_pred = model(batched_graph, target_poly, circuit_actions_batch, mask)
            pred_action = torch.argmax(action_logits[0]).item()
            if pred_action == action: action_correct += 1
            value_mse += F.mse_loss(value_pred, value_tensor, reduction='sum').item()
    action_accuracy = 100 * action_correct / len(test_indices)
    avg_value_mse = value_mse / len(test_indices)
    print(f"  Test Set Action Accuracy: {action_accuracy:.2f}%")
    print(f"  Test Set Value MSE: {avg_value_mse:.4f}")
    print(f"--------------------------------------------------\n")
    return action_accuracy, avg_value_mse

# --- main (Focuses on Supervised First) ---
def main():
    config = Config()
    n = config.n_variables
    d = config.max_complexity * 2 # Degree needs to be sufficient
    # Generate monomial indexing based on the *simplified* config
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)
    max_vector_size = max(monomial_to_index.values()) + 1

    # Create separate train and test datasets using the *simplified* config
    train_dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, config, size=config.train_size, description="Training")
    test_dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, config, size=config.test_size, description="Testing")

    # Initialize a *new* (or re-initialize) model with the *simplified* config
    model = CircuitBuilder(config, max_vector_size).to(device)
    best_model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
    ppo_model_path = f"ppo_model_n{config.n_variables}_C{config.max_complexity}.pt"

    # --- Focus on Supervised Training ---
    print("\n--- Starting/Resuming Supervised Training with Simplified Config ---")
    # Optionally load if a *best* supervised model exists for this config
    # if os.path.exists(best_model_path):
    #     print(f"Loading existing best supervised model from {best_model_path}")
    #     model.load_state_dict(torch.load(best_model_path, map_location=device))

    model = train_supervised(model, train_dataset, test_dataset, config)

    print("\n--- Final Supervised Evaluation ---")
    final_acc, _ = evaluate_model(model, test_dataset, config)

    # --- Conditional PPO Training ---
    if final_acc > 10.0: # Only start PPO if supervised accuracy is somewhat reasonable (e.g., > 10%)
        print("\nSupervised accuracy is above threshold, starting PPO training...")
        train_ppo(model, train_dataset, config)
        print(f"Saving final PPO model to {ppo_model_path}")
        torch.save(model.state_dict(), ppo_model_path)
    else:
        print(f"\nSupervised accuracy ({final_acc:.2f}%) is too low. PPO training skipped.")
        print("Focus on improving supervised learning. Try further simplification or architectural changes.")

    print("\nScript finished.")

if __name__ == "__main__":
    main()