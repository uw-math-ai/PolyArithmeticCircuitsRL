import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from PositionalEncoding import *
from State import Game
from encoders.compact_encoder import CompactOneHotGraphEncoder
from generator import generate_random_circuit, generate_random_polynomials
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from utils import encode_action

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- Configuration ---
class Config:
    def __init__(self):
        # --- Simplified Settings (Adjust as needed) ---
        self.n_variables = 3
        self.max_complexity = 5  # Max operations allowed
        self.max_degree = self.max_complexity * 2 # Max degree of polynomials
        self.hidden_dim = 256
        self.embedding_dim = 256
        self.num_gnn_layers = 3
        self.num_transformer_layers = 6
        self.transformer_heads = 4
        self.transformer_dropout = 0.1
        self.train_size = 10000
        self.test_size = 2000
        self.epochs = 50
        # --- Standard Settings ---
        self.learning_rate = 0.0003
        self.batch_size = 128
        self.mod = 50
        self.max_circuit_length = 100
        self.weight_decay = 0.01

        # --- PPO Hyperparameters ---
        self.rl_learning_rate = 1e-5
        self.ppo_iterations = 2000
        self.steps_per_batch = 4096
        self.ppo_epochs = 10
        self.ppo_minibatch_size = 128
        self.ppo_clip = 0.2
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.vf_coef = 0.5
        self.ent_coef = 0.02
        self.rl_eps = 1e-8
        self.action_temperature = 1.0
        self.step_penalty = -1.0
        self.success_reward = 100.0

        # --- Curriculum Learning ---
        self.complexity_threshold = 0.6
        self.complexity_window = 200
        self.compact_size = CompactOneHotGraphEncoder(
            N=self.max_complexity,
            P=self.mod,
            D=self.n_variables,
        ).size

        # --- Interesting polynomial data ---
        self.use_interesting_polynomials = True
        self.interesting_data_dir = PROJECT_ROOT / "Game-Board-Generation" / "pre-training-data"
        self.interesting_prefix = "game_board_C4"
        self.interesting_analysis_file = "path_analysis.jsonl"
        self.max_interesting_samples = 5000
        self.interesting_only_multipath = True

config = Config()

def build_compact_encoder(config: Config) -> CompactOneHotGraphEncoder:
    """Helper to instantiate a fresh compact encoder for this config."""
    return CompactOneHotGraphEncoder(
        N=config.max_complexity,
        P=config.mod,
        D=config.n_variables,
    )


def encode_actions_with_compact_encoder(actions, config: Config) -> torch.Tensor:
    """
    Encode a sequence of circuit actions using the compact one-hot encoder.
    Inputs/constant actions are ignored; only arithmetic ops update the encoding.
    """
    encoder = build_compact_encoder(config)
    for action_type, node1_id, node2_id in actions:
        if action_type not in ("add", "multiply"):
            continue
        op_type = 0 if action_type == "add" else 1
        encoder.update(node1_id, node2_id, op_type)
    encoding = encoder.get_encoding().copy()
    return torch.from_numpy(encoding)


INTERESTING_CACHE = None


def load_interesting_circuit_data(config: Config):
    """Load interesting polynomials + reconstructed circuits from JSONL dumps."""
    global INTERESTING_CACHE

    if not config.use_interesting_polynomials:
        INTERESTING_CACHE = []
        return INTERESTING_CACHE

    if INTERESTING_CACHE is not None:
        return INTERESTING_CACHE

    base_dir = config.interesting_data_dir
    nodes_path = base_dir / f"{config.interesting_prefix}.nodes.jsonl"
    edges_path = base_dir / f"{config.interesting_prefix}.edges.jsonl"
    analysis_path = base_dir / config.interesting_analysis_file
    missing = [p for p in (nodes_path, edges_path, analysis_path) if not p.exists()]

    if missing:
        print(
            "Warning: Missing interesting polynomial files: "
            + ", ".join(str(p) for p in missing)
            + ". Falling back to randomly generated circuits."
        )
        INTERESTING_CACHE = []
        return INTERESTING_CACHE

    with nodes_path.open("r", encoding="utf-8") as handle:
        node_exprs = {}
        node_steps = {}
        base_symbol_ids = []
        constant_node_ids = set()
        for line in handle:
            record = json.loads(line)
            node_id = record["id"]
            expr_str = record.get("expr_str") or record.get("label") or node_id
            expr = sympy.sympify(expr_str)
            node_exprs[node_id] = expr
            node_steps[node_id] = record.get("step", 0)
            if record.get("step", 0) == 0 and isinstance(expr, sympy.Symbol):
                base_symbol_ids.append(node_id)
            if expr == sympy.Integer(1):
                constant_node_ids.add(node_id)

    with analysis_path.open("r", encoding="utf-8") as handle:
        interesting_nodes = []
        for line in handle:
            record = json.loads(line)
            shortest_length = record.get("shortest_length")
            if shortest_length is None:
                continue
            if shortest_length > config.max_complexity:
                continue
            if config.interesting_only_multipath and not (
                record.get("multiple_shortest_paths") or record.get("multiple_paths")
            ):
                continue
            interesting_nodes.append((record["id"], shortest_length))

    with edges_path.open("r", encoding="utf-8") as handle:
        operations = {}
        dedup = set()
        for line in handle:
            record = json.loads(line)
            source = record.get("source")
            target = record.get("target")
            operand = record.get("operand")
            op = record.get("op")
            if not source or not target or operand is None:
                continue
            if op not in ("add", "mul"):
                continue
            ordered = tuple(sorted([source, operand]))
            key = (target, op, ordered)
            if key in dedup:
                continue
            dedup.add(key)
            operations.setdefault(target, []).append((op, ordered[0], ordered[1]))

    if len(base_symbol_ids) > config.n_variables:
        raise ValueError(
            f"Interesting polynomial data was generated with {len(base_symbol_ids)} variables "
            f"but config only provisions {config.n_variables}. Please update Config.n_variables."
        )

    def build_actions_for_target(target_id: str):
        actions = []
        node_to_idx = {}

        for input_idx in range(config.n_variables):
            actions.append(("input", input_idx, -1))
        for offset, node_id in enumerate(sorted(base_symbol_ids)):
            node_to_idx[node_id] = offset

        constant_idx = len(actions)
        actions.append(("constant", -1, -1))
        for node_id in constant_node_ids:
            node_to_idx[node_id] = constant_idx

        visiting = set()

        def ensure_node(node_id: str):
            if node_id in node_to_idx:
                return node_to_idx[node_id]
            if node_id in visiting:
                raise RuntimeError(f"Cyclic dependency detected for {node_id}")

            ops = operations.get(node_id, [])
            if not ops:
                raise KeyError(node_id)

            visiting.add(node_id)
            ops_sorted = sorted(
                ops,
                key=lambda entry: (
                    max(
                        node_steps.get(entry[1], math.inf),
                        node_steps.get(entry[2], math.inf),
                    ),
                    entry[0],
                    entry[1],
                    entry[2],
                ),
            )
            for op_type, left_id, right_id in ops_sorted:
                try:
                    left_idx = ensure_node(left_id)
                    right_idx = ensure_node(right_id)
                except KeyError:
                    continue

                node_index = len(actions)
                action_type = "add" if op_type == "add" else "multiply"
                actions.append((action_type, left_idx, right_idx))
                node_to_idx[node_id] = node_index
                visiting.remove(node_id)
                return node_index

            visiting.remove(node_id)
            raise KeyError(node_id)

        ensure_node(target_id)
        return actions

    interesting_data = []
    failures = 0
    seen = set()
    for node_id, shortest_length in interesting_nodes:
        if node_id in seen or node_id not in node_exprs:
            continue
        try:
            actions = build_actions_for_target(node_id)
        except (KeyError, RuntimeError):
            failures += 1
            continue

        encoding = encode_actions_with_compact_encoder(actions, config).clone()
        interesting_data.append(
            {
                "id": node_id,
                "expr": node_exprs[node_id],
                "actions": tuple(actions),
                "encoding": encoding,
                "shortest_length": shortest_length,
            }
        )
        seen.add(node_id)
        if len(interesting_data) >= config.max_interesting_samples:
            break

    if interesting_data:
        print(
            f"Loaded {len(interesting_data)} interesting polynomials from {analysis_path} "
            f"(skipped {failures} nodes)."
        )

    INTERESTING_CACHE = interesting_data
    return INTERESTING_CACHE

# --- GNN Model ---
class ArithmeticCircuitGNN(nn.Module):
    """GNN to embed the current arithmetic circuit state."""

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(ArithmeticCircuitGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(config.num_gnn_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(config.num_gnn_layers - 1)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0 and x.size(0) > 0:
            return torch.zeros(x.size(0), self.convs[-1].out_channels, device=x.device)
        elif x.size(0) == 0:
            return torch.zeros(0, self.convs[-1].out_channels, device=x.device)

        x = F.relu(self.convs[0](x, edge_index))
        for i in range(1, len(self.convs) - 1):
            identity = x
            x = self.layer_norms[i - 1](x)
            x = F.relu(self.convs[i](x, edge_index))
            x = x + identity
        x = self.convs[-1](x, edge_index)
        x = self.final_norm(x)
        return x


# --- Main Model (Transformer + GNN) ---
class CircuitBuilder(nn.Module):
    """Main model combining GNN, Transformer, Policy, and Value heads."""

    def __init__(self, config, state_encoding_size):
        super(CircuitBuilder, self).__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.gnn = ArithmeticCircuitGNN(
            4, config.hidden_dim, config.embedding_dim
        )
        self.polynomial_embedding = nn.Linear(
            state_encoding_size, config.embedding_dim
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.transformer_dropout,
            batch_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, config.num_transformer_layers
        )
        max_nodes = config.n_variables + config.max_complexity + 1
        max_actions = (max_nodes * (max_nodes + 1) // 2) * 2
        self.action_head = nn.Linear(config.embedding_dim, max_actions)
        self.value_head = nn.Linear(config.embedding_dim, 1)
        self.output_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))

    def forward(
        self,
        batched_graph,
        target_polynomials,
        circuit_actions,
        available_actions_masks=None,
    ):
        batch_size = target_polynomials.size(0)

        # Get node embeddings from GNN
        node_embeddings = self.gnn(batched_graph)
        
        # Aggregate node embeddings into a single graph embedding per graph in the batch
        # using global mean pooling
        graph_embeddings = global_mean_pool(node_embeddings, batched_graph.batch)
        # graph_embeddings shape: (batch_size, embedding_dim)
        
        # Embed the target polynomial
        poly_embeddings = self.polynomial_embedding(target_polynomials)
        # poly_embeddings shape: (batch_size, embedding_dim)

        # Build memory for transformer decoder:
        # memory should contain both the target (polynomial) and current state (graph)
        # Shape: (2, batch_size, embedding_dim)
        memory = torch.stack([poly_embeddings, graph_embeddings], dim=0)

        query = self.output_token.expand(-1, batch_size, -1)
        output = self.transformer_decoder(tgt=query, memory=memory)
        output_squeezed = output.squeeze(0)

        action_logits = self.action_head(output_squeezed)
        value_pred = self.value_head(output_squeezed).squeeze(-1)

        if available_actions_masks is not None:
            if action_logits.size(1) > available_actions_masks.size(1):
                padding = torch.zeros(
                    batch_size,
                    action_logits.size(1) - available_actions_masks.size(1),
                    dtype=torch.bool,
                    device=device,
                )
                available_actions_masks = torch.cat(
                    [available_actions_masks, padding], dim=1
                )
            elif action_logits.size(1) < available_actions_masks.size(1):
                available_actions_masks = available_actions_masks[
                    :, : action_logits.size(1)
                ]

            action_logits = action_logits.masked_fill(
                ~available_actions_masks, float("-inf")
            )

        return action_logits, value_pred

    def get_action_and_value(self, state, action_idx=None, temperature=1.0):
        circuit_graph, target_poly, circuit_actions, mask = state
        batched_graph = Batch.from_data_list([circuit_graph.to(device)])
        target_poly = target_poly.to(device)
        mask = mask.to(device)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)

        action_logits, value_pred = self.forward(
            batched_graph, target_poly, circuit_actions, mask
        )

        valid_indices = torch.where(mask[0])[0]
        if len(valid_indices) == 0:
            print("Warning: No valid actions available!")
            return None, None, None, value_pred

        valid_logits = action_logits[0, valid_indices]
        dist = Categorical(logits=valid_logits / temperature)

        if action_idx is None:
            local_action = dist.sample()
            action = valid_indices[local_action].item()
        else:
            action = action_idx
            local_action_tensor = (valid_indices == action).nonzero(as_tuple=True)[0]
            if len(local_action_tensor) == 0:
                print(f"Warning: Action {action} not found in valid indices {valid_indices}. Logits: {action_logits}")
                return None, None, None, value_pred
            local_action = local_action_tensor[0]

        log_prob = dist.log_prob(local_action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value_pred


# --- Dataset ---
class CircuitDataset(Dataset):
    """Generates (state, next_action) pairs for supervised learning."""

    def __init__(
        self,
        config,
        size=10000,
        description="Training",
    ):
        self.config = config
        print(f"Generating {description} dataset ({size} examples) with N={config.n_variables}, C={config.max_complexity}...")
        self.data = self.generate_data(size)
        print(f"Finished generating {description} dataset.")

    def generate_data(self, size):
        dataset = []
        n = self.config.n_variables
        C = self.config.max_complexity
        num_circuits = size // C if C > 0 else size

        interesting_entries = load_interesting_circuit_data(self.config)
        if interesting_entries:
            print(
                f"Using {len(interesting_entries)} interesting polynomials for supervised data."
            )
            while len(dataset) < size:
                entry = random.choice(interesting_entries)
                target_encoding = entry["encoding"]
                finished = self._add_examples_from_actions(
                    dataset,
                    list(entry["actions"]),
                    target_encoding,
                    size_limit=size,
                )
                if finished:
                    break
            return dataset[:size]

        _, all_circuits, _ = generate_random_polynomials(
            n, C, num_polynomials=num_circuits, mod=self.config.mod
        )

        for i in tqdm.tqdm(range(len(all_circuits)), desc=f"Processing generated data"):
            actions = all_circuits[i]
            target_encoding = encode_actions_with_compact_encoder(actions, self.config)
            finished = self._add_examples_from_actions(
                dataset, actions, target_encoding, size_limit=size
            )
            if finished:
                return dataset
        return dataset

    def _add_examples_from_actions(
        self, dataset, actions, target_encoding, size_limit=None
    ):
        actions_list = list(actions)
        n_base = self.config.n_variables + 1
        max_nodes = self.config.n_variables + self.config.max_complexity + 1

        for i in range(n_base, len(actions_list)):
            current_actions = actions_list[:i]
            next_action = actions_list[i]
            next_op, next_node1_id, next_node2_id = next_action

            available_mask = self.get_available_actions_mask(
                current_actions, max_nodes
            )
            action_idx = encode_action(next_op, next_node1_id, next_node2_id, max_nodes)

            if action_idx >= len(available_mask):
                continue

            dataset.append(
                {
                    "actions": current_actions,
                    "target_poly": target_encoding.clone(),
                    "mask": available_mask.cpu(),
                    "action": action_idx,
                    "value": (len(actions_list) - i)
                    / max(1, len(actions_list) - n_base),
                }
            )
            if size_limit is not None and len(dataset) >= size_limit:
                return True
        return False

    def get_available_actions_mask(self, actions, max_nodes):
        n_nodes = len(actions)
        total_max_pairs = (max_nodes * (max_nodes + 1)) // 2
        max_possible_actions = total_max_pairs * 2
        mask = torch.zeros(max_possible_actions, dtype=torch.bool)
        for i in range(n_nodes):
            for j in range(i, n_nodes):
                for op in ["add", "multiply"]:
                    action_idx = encode_action(op, i, j, max_nodes)
                    if action_idx < max_possible_actions:
                        mask[action_idx] = True
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        circuit_graph = self.actions_to_graph(item["actions"])
        target_poly = item["target_poly"]

        return (
            circuit_graph,
            target_poly,
            item["actions"],
            item["mask"],
            item["action"],
            item["value"],
        )

    def actions_to_graph(self, actions):
        n_nodes = len(actions)
        node_features, edges = [], []
        for i, (action_type, input1_idx, input2_idx) in enumerate(actions):
            if action_type == "input":
                type_encoding, value = [1, 0, 0], i / max(1, self.config.n_variables)
            elif action_type == "constant":
                type_encoding, value = [0, 1, 0], 1.0
            else:
                type_encoding, value = (
                    [0, 0, 1],
                    1.0 if action_type == "multiply" else 0.0,
                )
                edges.append((input1_idx, i))
                edges.append((input2_idx, i))
            node_features.append(type_encoding + [value])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_nodes)
        return Data(x=x, edge_index=edge_index)


# --- Collate Function ---
def circuit_collate(batch):
    (
        graphs,
        target_polys_list,
        circuit_actions,
        masks_list,
        actions_list,
        values_list,
    ) = zip(*batch)
    batched_graph = Batch.from_data_list(graphs).to(device)
    target_polys = torch.stack(target_polys_list).to(device)
    max_mask_len = max(m.size(0) for m in masks_list)
    padded_masks = []
    for m in masks_list:
        padding = torch.zeros(max_mask_len - m.size(0), dtype=torch.bool)
        padded_masks.append(torch.cat([m, padding]))
    masks = torch.stack(padded_masks).to(device)
    actions = torch.tensor(actions_list, device=device)
    values = torch.tensor(values_list, dtype=torch.float, device=device)
    return batched_graph, target_polys, list(circuit_actions), masks, actions, values


# --- Supervised Training ---
def train_supervised(model, train_dataset, test_dataset, config):
    data_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=circuit_collate,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(data_loader) * config.epochs,
        eta_min=config.learning_rate / 100,
    )
    best_test_acc = 0.0

    print("\n--- Starting Supervised Training ---")
    for epoch in range(config.epochs):
        model.train()
        total_action_loss, total_value_loss, action_correct, total = 0, 0, 0, 0
        for (
            batched_graph,
            target_polys,
            circuit_actions,
            masks,
            actions,
            values,
        ) in tqdm.tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config.epochs}"):
            optimizer.zero_grad()
            action_logits, value_preds = model(
                batched_graph, target_polys, circuit_actions, masks
            )
            action_loss = F.cross_entropy(action_logits, actions)
            value_loss = F.mse_loss(value_preds, values)
            loss = action_loss + config.vf_coef * value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            action_correct += (
                (torch.argmax(action_logits, dim=1) == actions).sum().item()
            )
            total += actions.size(0)
            total_action_loss += action_loss.item() * actions.size(0)
            total_value_loss += value_loss.item() * actions.size(0)

        train_acc = 100 * action_correct / total
        print(
            f"Epoch {epoch + 1}: LR: {optimizer.param_groups[0]['lr']:.6f}, Train Acc: {train_acc:.2f}%, Loss: {total_action_loss / total:.4f}, VLoss: {total_value_loss / total:.4f}"
        )

        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            test_acc, _ = evaluate_model(model, test_dataset, config)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                model_path = f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
                torch.save(model.state_dict(), model_path)
                print(
                    f"*** New Best Test Accuracy ({best_test_acc:.2f}%)! Model saved to {model_path} ***"
                )
    print(f"Finished Supervised Training. Best Test Accuracy: {best_test_acc:.2f}%")
    return model


# --- PPO ---
def calculate_gae(rewards, values, dones, gamma, lambda_gae):
    advantages = []
    gae = 0.0
    values_ext = values + [values[-1] if not dones[-1] else 0.0]
    for i in reversed(range(len(rewards))):
        delta = (
            rewards[i] + gamma * values_ext[i + 1] * (1.0 - dones[i]) - values_ext[i]
        )
        gae = delta + gamma * lambda_gae * (1.0 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values)]
    return torch.tensor(advantages, dtype=torch.float, device=device), torch.tensor(
        returns, dtype=torch.float, device=device
    )


def train_ppo(model, config):
    print("\n--- Starting PPO Training ---")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.rl_learning_rate, eps=config.rl_eps
    )

    interesting_entries = load_interesting_circuit_data(config)
    if interesting_entries:
        print(
            f"PPO will sample targets from {len(interesting_entries)} interesting polynomials."
        )
    
    current_complexity = 1
    recent_successes = []

    for ppo_iter in range(config.ppo_iterations):
        model.eval()
        batch_states, batch_actions, batch_old_log_probs = [], [], []
        batch_rewards, batch_dones, batch_values = [], [], []
        collected_steps, games_played, success_count = 0, 0, 0
        circuit_examples = []
        iteration_successes = []

        with torch.no_grad():
            pbar = tqdm.tqdm(
                total=config.steps_per_batch,
                desc=f"PPO Iter {ppo_iter + 1} (C={current_complexity}) - Collecting",
            )
            while collected_steps < config.steps_per_batch:
                games_played += 1
                n = config.n_variables

                if interesting_entries:
                    eligible = [
                        entry
                        for entry in interesting_entries
                        if entry["shortest_length"] <= current_complexity
                    ]
                    pool = eligible if eligible else interesting_entries
                    entry = random.choice(pool)
                    target_poly_expr = entry["expr"]
                    target_poly_encoding = entry["encoding"].clone().unsqueeze(0)
                else:
                    actions_gen, polynomials_gen = generate_random_circuit(
                        n, current_complexity, mod=config.mod
                    )
                    if not polynomials_gen:
                        continue

                    target_poly_expr = polynomials_gen[-1]
                    target_poly_encoding = encode_actions_with_compact_encoder(
                        actions_gen, config
                    ).unsqueeze(0)

                game = Game(
                    target_poly_expr,
                    target_poly_encoding,
                    config,
                ).to("cpu")
                
                traj_states, traj_actions, traj_log_probs, traj_rewards, traj_dones, traj_values = [], [], [], [], [], []

                while not game.is_done():
                    state_tuple = game.observe()
                    action, log_prob, _, value = model.get_action_and_value(
                        state_tuple, temperature=config.action_temperature
                    )

                    if action is None:
                        print("Warning: Action selection failed, ending game.")
                        break

                    game.take_action(action)
                    rewards = game.compute_rewards()
                    reward = rewards[-1] if rewards else 0.0
                    done = game.is_done()

                    traj_states.append(state_tuple)
                    traj_actions.append(action)
                    traj_log_probs.append(log_prob.cpu())
                    traj_values.append(value.cpu().item())
                    traj_rewards.append(reward)
                    traj_dones.append(done)

                if traj_states:
                    batch_states.extend(traj_states)
                    batch_actions.extend(traj_actions)
                    batch_old_log_probs.extend(traj_log_probs)
                    batch_values.extend(traj_values)
                    batch_rewards.extend(traj_rewards)
                    batch_dones.extend(traj_dones)
                    collected_steps += len(traj_states)
                    pbar.update(len(traj_states))

                    success = traj_dones[-1] and traj_rewards[-1] > 5.0
                    if success:
                        success_count += 1
                    iteration_successes.append(success)

                    if len(circuit_examples) < 5:
                        circuit_examples.append(
                            {
                                "target": target_poly_expr,
                                "success": success,
                                "reward": traj_rewards[-1],
                                "steps": len(traj_states),
                            }
                        )
            pbar.close()

        recent_successes.extend(iteration_successes)
        recent_successes = recent_successes[-config.complexity_window:]
        if len(recent_successes) >= config.complexity_window // 2:
            recent_success_rate = sum(recent_successes) / len(recent_successes)
            if (
                recent_success_rate > config.complexity_threshold
                and current_complexity < config.max_complexity
            ):
                current_complexity += 1
                recent_successes = []
                print(
                    f"*** Complexity Increased to {current_complexity} (SR: {recent_success_rate:.2f}) ***"
                )

        if not batch_states:
            print(f"PPO Iter {ppo_iter + 1}: No data collected, skipping update.")
            continue

        advantages, returns = calculate_gae(
            batch_rewards, batch_values, batch_dones, config.gamma, config.lambda_gae
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + config.rl_eps
        )

        batch_old_log_probs_t = torch.stack(batch_old_log_probs).to(device)
        batch_actions_t = torch.tensor(batch_actions, device=device, dtype=torch.long)

        model.train()
        indices = np.arange(len(batch_states))

        for ppo_epoch in range(config.ppo_epochs):
            np.random.shuffle(indices)
            total_pi_loss, total_v_loss, total_ent_loss = 0, 0, 0
            num_updates = 0

            for i in range(0, len(indices), config.ppo_minibatch_size):
                mb_indices = indices[i : i + config.ppo_minibatch_size]
                mb_states = [batch_states[idx] for idx in mb_indices]
                mb_actions = batch_actions_t[mb_indices]
                mb_old_log_probs = batch_old_log_probs_t[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                mb_graphs = [s[0] for s in mb_states]
                mb_polys = torch.cat([s[1] for s in mb_states]).to(device)
                mb_circ_actions = [s[2][0] for s in mb_states]
                mb_masks = torch.cat([s[3] for s in mb_states]).to(device)

                mb_batched_graph = Batch.from_data_list(mb_graphs).to(device)

                action_logits, values_pred = model(
                    mb_batched_graph, mb_polys, mb_circ_actions, mb_masks
                )

                new_log_probs_list, entropies_list = [], []
                for j, logits in enumerate(action_logits):
                    mask_j = mb_masks[j]
                    valid_indices = torch.where(mask_j)[0]

                    if len(valid_indices) == 0:
                        continue

                    valid_logits = logits[valid_indices]
                    dist = Categorical(logits=valid_logits)
                    action = mb_actions[j].item()
                    local_action_tensor = (valid_indices == action).nonzero(as_tuple=True)[0]

                    if len(local_action_tensor) == 0:
                        full_dist = Categorical(logits=logits)
                        new_log_probs_list.append(full_dist.log_prob(mb_actions[j]))
                        entropies_list.append(full_dist.entropy())
                    else:
                        local_action = local_action_tensor[0]
                        new_log_probs_list.append(dist.log_prob(local_action))
                        entropies_list.append(dist.entropy())

                if not new_log_probs_list:
                    continue

                new_log_probs_t = torch.stack(new_log_probs_list)
                entropies_t = torch.stack(entropies_list)

                ratios = torch.exp(new_log_probs_t - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - config.ppo_clip, 1 + config.ppo_clip)
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values_pred, mb_returns)
                entropy_loss = -entropies_t.mean()
                loss = (
                    policy_loss
                    + config.vf_coef * value_loss
                    + config.ent_coef * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_pi_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_ent_loss += entropy_loss.item()
                num_updates += 1

        avg_pi_loss = total_pi_loss / num_updates if num_updates > 0 else 0
        avg_v_loss = total_v_loss / num_updates if num_updates > 0 else 0
        avg_ent_loss = total_ent_loss / num_updates if num_updates > 0 else 0
        success_rate = 100 * success_count / games_played if games_played > 0 else 0
        recent_sr = (
            100 * sum(recent_successes) / len(recent_successes)
            if recent_successes
            else 0
        )
        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0

        print(
            f"\nPPO Iter {ppo_iter + 1}: SR: {success_rate:.1f}% (Recent: {recent_sr:.1f}%), C: {current_complexity}, "
            f"Reward: {avg_reward:.3f}, PiL: {avg_pi_loss:.4f}, VL: {avg_v_loss:.4f}, Ent: {-avg_ent_loss:.4f}"
        )
        for i, ex in enumerate(circuit_examples):
            print(
                f"  Ex {i + 1}: {'Success' if ex['success'] else 'Fail'} (R: {ex['reward']:.2f}, S: {ex['steps']}) Target: {ex['target']}"
            )

        if (ppo_iter + 1) % 50 == 0:
            path = f"ppo_model_n{config.n_variables}_C{config.max_complexity}_curriculum.pt"
            torch.save(model.state_dict(), path)
            print(f"  Model saved to {path}")


# --- Evaluation ---
def evaluate_model(model, test_dataset, config, num_tests=100):
    """Evaluates the model's supervised prediction accuracy."""
    model.eval()
    print(
        f"\n--- Evaluating Supervised Performance on {min(num_tests, len(test_dataset))} Test Examples ---"
    )
    test_indices = random.sample(
        range(len(test_dataset)), min(num_tests, len(test_dataset))
    )
    action_correct = 0
    with torch.no_grad():
        for idx in test_indices:
            circuit_graph, target_poly, circuit_actions, mask, action, value = (
                test_dataset[idx]
            )
            batched_graph = Batch.from_data_list([circuit_graph]).to(device)
            target_poly = target_poly.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            action_logits, _ = model(
                batched_graph, target_poly, [circuit_actions], mask
            )
            pred_action = torch.argmax(action_logits[0]).item()
            if pred_action == action:
                action_correct += 1
    action_accuracy = 100 * action_correct / len(test_indices)
    print(f"  Test Set Action Accuracy: {action_accuracy:.2f}%")
    print(f"--------------------------------------------------\n")
    return action_accuracy, 0


# --- Main Execution ---
def main():
    config = Config()
    print(f"Compact encoder size: {config.compact_size}")

    # Create datasets
    train_dataset = CircuitDataset(
        config,
        size=config.train_size,
        description="Training",
    )
    test_dataset = CircuitDataset(
        config,
        size=config.test_size,
        description="Testing",
    )

    # Initialize model
    model = CircuitBuilder(config, config.compact_size).to(device)
    best_model_path = (
        f"best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
    )
    ppo_model_path = (
        f"ppo_model_n{config.n_variables}_C{config.max_complexity}_curriculum.pt"
    )

    # Load or train supervised model
    if os.path.exists(best_model_path):
        print(f"Loading pretrained supervised model from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("\n--- No supervised model found. Starting Supervised Training ---")
        model = train_supervised(model, train_dataset, test_dataset, config)

    # Evaluate supervised model
    final_acc, _ = evaluate_model(model, test_dataset, config)

    # Conditionally start PPO Training
    if final_acc > 15.0 or os.path.exists(ppo_model_path):
        if os.path.exists(ppo_model_path):
            print(f"\nLoading existing PPO model from {ppo_model_path}")
            model.load_state_dict(torch.load(ppo_model_path, map_location=device))
        print("\n--- Starting/Resuming PPO Training ---")
        train_ppo(model, config)
        print(f"Saving final PPO model to {ppo_model_path}")
        torch.save(model.state_dict(), ppo_model_path)
    else:
        print(
            f"\nSupervised accuracy ({final_acc:.2f}%) is below 15%. PPO training skipped."
        )
        print("Consider improving supervised learning or lowering the threshold.")

    print("\nScript finished.")


if __name__ == "__main__":
    main()
