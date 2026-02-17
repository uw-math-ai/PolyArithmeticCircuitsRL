import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = SRC_ROOT.parent
PPO_DIR = SRC_ROOT / "PPO RL"
for path in (CURRENT_DIR, SRC_ROOT, PROJECT_ROOT, PPO_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import numpy as np
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from torch.distributions import Categorical
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import add_self_loops

from State import Game
from encoders.compact_encoder import CompactOneHotGraphEncoder
from generator import generate_random_circuit
from mcts import MCTSPlanner
from utils import encode_action


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Config:
    """Configuration for discrete SAC with optional MCTS guidance."""

    def __init__(self):
        self.n_variables = 3
        self.max_complexity = 8
        self.max_degree = self.max_complexity * 2
        self.hidden_dim = 256
        self.embedding_dim = 256
        self.num_gnn_layers = 3
        self.num_transformer_layers = 6
        self.transformer_heads = 4
        self.transformer_dropout = 0.1
        self.mod = 50

        # SAC hyperparameters
        self.gamma = 0.99
        self.alpha = 0.2
        self.tau = 0.005
        self.learning_rate = 3e-4
        self.batch_size = 128
        self.buffer_size = 200000
        self.min_buffer_size = 2000
        self.steps_per_iter = 4096
        self.updates_per_iter = 256
        self.action_temperature = 1.2
        self.rl_eps = 1e-8
        self.step_penalty = -0.05
        self.success_reward = 100.0
        self.model_tag = "v4"
        self.resume = True
        self.resume_path = f"sac_model_v3_n{self.n_variables}_C{self.max_complexity}.pt"

        # MCTS guidance
        self.use_mcts = True
        self.mcts_simulations = 96
        self.mcts_exploration = 1.4
        self.mcts_policy_mix = 0.5
        self.mcts_policy_temperature = 1.0
        self.mcts_ce_coef = 0.5

        # Curriculum learning
        self.complexity_threshold = 0.6
        self.complexity_window = 300
        self.sr_advance_threshold = 0.75
        self.sr_backoff_threshold = 0.55
        self.allow_complexity_backoff = True

        # Interesting polynomial data
        self.use_interesting_polynomials = False
        self.interesting_data_dir = PROJECT_ROOT / "Game-Board-Generation" / "pre-training-data"
        self.interesting_prefix = "game_board_C1"
        self.interesting_analysis_file = "game_board_C1.analysis.jsonl"
        self.max_interesting_samples = 5000
        self.interesting_only_multipath = True
        # "dataset" -> only precomputed interesting polynomials
        # "mixed" -> OpenTensor-style pattern + random sampling (per-episode)
        # "pool" -> precompute a mixed pool once per complexity, then sample
        # "random" -> random circuits only
        self.training_target_mode = "random"
        self.target_pool_size = 2000
        self.pool_interesting_ratio = 0.6
        self.pool_pattern_ratio = 0.25
        self.pool_random_ratio = 0.15

        # Synthetic dataset (OpenTensor-style) prefill
        self.use_synthetic_dataset = True
        self.synthetic_samples = 5000
        self.synthetic_complexity_min = 1
        self.synthetic_complexity_max = None

        # Logging verbosity
        self.show_progress_bars = False

        # Weights & Biases logging
        self.use_wandb = False
        self.wandb_project = "SAC RL-MCTS"
        self.wandb_run_name = None

        self.compact_size = CompactOneHotGraphEncoder(
            N=self.max_complexity,
            P=self.mod,
            D=self.n_variables,
        ).size

    @property
    def max_nodes(self) -> int:
        return self.n_variables + self.max_complexity + 1

    @property
    def max_actions(self) -> int:
        total_max_pairs = (self.max_nodes * (self.max_nodes + 1)) // 2
        return total_max_pairs * 2


config = Config()


def build_compact_encoder(config: Config) -> CompactOneHotGraphEncoder:
    return CompactOneHotGraphEncoder(
        N=config.max_complexity,
        P=config.mod,
        D=config.n_variables,
    )


def encode_actions_with_compact_encoder(actions, config: Config) -> torch.Tensor:
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

        op_count = sum(1 for op, _, _ in actions if op in ("add", "multiply"))
        if op_count > config.max_complexity:
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


class ArithmeticCircuitGNN(nn.Module):
    """GNN to embed the current arithmetic circuit state."""

    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, embedding_dim))
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if edge_index.numel() == 0 and x.size(0) > 0:
            return torch.zeros(x.size(0), self.convs[-1].out_channels, device=x.device)
        if x.size(0) == 0:
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


class SACCircuitBuilder(nn.Module):
    """Shared encoder with policy logits and twin Q heads."""

    def __init__(self, config: Config, state_encoding_size: int):
        super().__init__()
        self.config = config
        self.gnn = ArithmeticCircuitGNN(
            4, config.hidden_dim, config.embedding_dim, config.num_gnn_layers
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
        self.action_head = nn.Linear(config.embedding_dim, config.max_actions)
        self.q1_head = nn.Linear(config.embedding_dim, config.max_actions)
        self.q2_head = nn.Linear(config.embedding_dim, config.max_actions)
        self.output_token = nn.Parameter(torch.randn(1, 1, config.embedding_dim))

    def forward(
        self,
        batched_graph: Batch,
        target_polynomials: torch.Tensor,
        available_actions_masks=None,
    ):
        batch_size = target_polynomials.size(0)

        node_embeddings = self.gnn(batched_graph)
        graph_embeddings = global_mean_pool(node_embeddings, batched_graph.batch)
        poly_embeddings = self.polynomial_embedding(target_polynomials)
        memory = torch.stack([poly_embeddings, graph_embeddings], dim=0)
        query = self.output_token.expand(-1, batch_size, -1)
        output = self.transformer_decoder(tgt=query, memory=memory)
        output_squeezed = output.squeeze(0)

        action_logits = self.action_head(output_squeezed)
        q1 = self.q1_head(output_squeezed)
        q2 = self.q2_head(output_squeezed)

        if available_actions_masks is not None:
            if action_logits.size(1) > available_actions_masks.size(1):
                padding = torch.zeros(
                    batch_size,
                    action_logits.size(1) - available_actions_masks.size(1),
                    dtype=torch.bool,
                    device=action_logits.device,
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

        return action_logits, q1, q2


class ReplayBuffer:
    """Simple replay buffer for off-policy SAC."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage = []
        self.pos = 0

    def __len__(self):
        return len(self.storage)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        mcts_pi,
        has_mcts,
    ):
        data = (state, action, reward, next_state, done, mcts_pi, has_mcts)
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        indices = random.sample(range(len(self.storage)), batch_size)
        return [self.storage[idx] for idx in indices]


def extract_state(state_tuple):
    circuit_graph, target_poly, _, mask = state_tuple
    target_poly = target_poly.squeeze(0).cpu()
    mask = mask.squeeze(0).cpu()
    return circuit_graph.to("cpu"), target_poly, mask


def policy_dict_to_vector(policy_dict, max_actions):
    pi = torch.zeros(max_actions, dtype=torch.float)
    for action, prob in policy_dict.items():
        if action < max_actions:
            pi[action] = float(prob)
    return pi


def actions_to_polynomials(actions, n_variables):
    symbols = [sympy.Symbol(f"x{i}") for i in range(n_variables)]
    polynomials = []
    for action_type, node1_id, node2_id in actions:
        if action_type == "input":
            polynomials.append(symbols[node1_id])
        elif action_type == "constant":
            polynomials.append(sympy.Integer(1))
        elif action_type == "add":
            polynomials.append(sympy.expand(polynomials[node1_id] + polynomials[node2_id]))
        elif action_type == "multiply":
            polynomials.append(sympy.expand(polynomials[node1_id] * polynomials[node2_id]))
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    return polynomials


def build_pattern_actions(n_variables, max_degree, complexity, max_attempts=20):
    rng = random.Random()
    symbols = [sympy.Symbol(f"x{i}") for i in range(n_variables)]

    def pick_vars(k):
        return [rng.choice(symbols) for _ in range(k)]

    def base_actions():
        actions = [("input", i, -1) for i in range(n_variables)]
        actions.append(("constant", -1, -1))
        return actions

    def add_op(actions, op, left, right):
        idx = len(actions)
        actions.append((op, left, right))
        return idx

    def var_index(var):
        return symbols.index(var)

    for _ in range(max_attempts):
        actions = base_actions()
        choice = None

        if complexity == 2:
            a, b, c = pick_vars(3)
            if rng.random() < 0.5:
                choice = "square_sum"
                s = add_op(actions, "add", var_index(a), var_index(b))
                add_op(actions, "multiply", s, s)
            else:
                choice = "distribute"
                s = add_op(actions, "add", var_index(b), var_index(c))
                add_op(actions, "multiply", var_index(a), s)
        elif complexity == 3:
            a, b, c, d = pick_vars(4)
            roll = rng.random()
            if roll < 0.4:
                choice = "square_sum3"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "add", s1, var_index(c))
                add_op(actions, "multiply", s2, s2)
            elif roll < 0.7:
                choice = "prod_sums2"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "add", var_index(c), var_index(d))
                add_op(actions, "multiply", s1, s2)
            else:
                choice = "a_times_square"
                s1 = add_op(actions, "add", var_index(b), var_index(c))
                s2 = add_op(actions, "multiply", s1, s1)
                add_op(actions, "multiply", var_index(a), s2)
        elif complexity == 4:
            a, b, c, d, e = pick_vars(5)
            roll = rng.random()
            if roll < 0.4:
                choice = "square_sum4"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "add", s1, var_index(c))
                s3 = add_op(actions, "add", s2, var_index(d))
                add_op(actions, "multiply", s3, s3)
            elif roll < 0.7:
                choice = "prod_sum3"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "add", var_index(c), var_index(d))
                s3 = add_op(actions, "add", s2, var_index(e))
                add_op(actions, "multiply", s1, s3)
            else:
                choice = "square_times_sum"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "multiply", s1, s1)
                s3 = add_op(actions, "add", var_index(c), var_index(d))
                add_op(actions, "multiply", s2, s3)
        elif complexity >= 5:
            a, b, c, d, e, f = pick_vars(6)
            roll = rng.random()
            if roll < 0.5:
                choice = "sum3_times_sum3"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "add", s1, var_index(c))
                s3 = add_op(actions, "add", var_index(d), var_index(e))
                s4 = add_op(actions, "add", s3, var_index(f))
                add_op(actions, "multiply", s2, s4)
            else:
                choice = "sum_sq_plus"
                s1 = add_op(actions, "add", var_index(a), var_index(b))
                s2 = add_op(actions, "multiply", s1, s1)
                s3 = add_op(actions, "add", var_index(c), var_index(d))
                s4 = add_op(actions, "multiply", s3, s3)
                add_op(actions, "add", s2, s4)
        else:
            continue

        try:
            polynomials = actions_to_polynomials(actions, n_variables)
        except Exception:
            continue
        target = polynomials[-1]
        if max_degree is not None:
            poly_obj = target.as_poly(*symbols)
            if poly_obj is None or poly_obj.total_degree() > max_degree:
                continue
        op_count = sum(1 for op, _, _ in actions if op in ("add", "multiply"))
        if op_count != complexity:
            continue
        return actions, target, choice

    return None, None, None


def generate_mixed_circuit(config: Config, complexity: int, seen_polynomials=None):
    actions, target_expr, _ = build_pattern_actions(
        config.n_variables, config.max_degree, complexity
    )
    if actions is None or target_expr is None:
        actions, polynomials = generate_random_circuit(
            config.n_variables, complexity, mod=config.mod
        )
        if not polynomials:
            return None, None, None
        target_expr = polynomials[-1]

    if seen_polynomials is not None:
        target_key = str(sympy.expand(target_expr))
        if target_key in seen_polynomials:
            return None, None, None
        seen_polynomials.add(target_key)

    encoding = encode_actions_with_compact_encoder(actions, config).unsqueeze(0)
    return actions, target_expr, encoding


def generate_synthetic_transitions(
    config: Config,
    samples_n: int,
    complexity_min: int = 1,
    complexity_max: Optional[int] = None,
):
    """
    OpenTensor-style synthetic dataset: generate targets from random circuits,
    then replay the exact circuit actions to build state-action-reward tuples.
    """
    transitions = []
    complexity_max = complexity_max or config.max_complexity
    base_nodes = config.n_variables + 1

    for _ in tqdm.tqdm(
        range(samples_n),
        desc="Generating synthetic dataset",
        disable=not config.show_progress_bars,
    ):
        complexity = random.randint(complexity_min, complexity_max)
        actions, polynomials = None, None
        for _ in range(20):
            actions, polynomials = generate_random_circuit(
                config.n_variables, complexity, mod=config.mod
            )
            if polynomials and len(actions or []) > base_nodes:
                break
        if not polynomials or len(actions or []) <= base_nodes:
            continue

        target_expr = polynomials[-1]
        target_encoding = encode_actions_with_compact_encoder(actions, config).unsqueeze(0)
        game = build_game_from_target(target_expr, target_encoding, config)

        for action in actions[base_nodes:]:
            state_tuple = game.observe()
            state = extract_state(state_tuple)

            op, node1_id, node2_id = action
            action_idx = encode_action(op, node1_id, node2_id, config.max_nodes)
            game.take_action(action_idx)

            rewards = game.compute_rewards()
            reward = rewards[-1] if rewards else 0.0
            done = game.is_done()

            next_state_tuple = game.observe()
            next_state = extract_state(next_state_tuple)

            transitions.append(
                (
                    state,
                    action_idx,
                    reward,
                    next_state,
                    float(done),
                    torch.zeros(config.max_actions, dtype=torch.float),
                    False,
                )
            )

            if done:
                break

    return transitions


def build_target_pool(
    config: Config,
    complexity: int,
    interesting_entries,
):
    pool = []
    seen = set()

    def add_target(expr, encoding):
        key = str(sympy.expand(expr))
        if key in seen:
            return False
        seen.add(key)
        pool.append((expr, encoding))
        return True

    pool_size = max(1, config.target_pool_size)
    interesting_n = int(pool_size * config.pool_interesting_ratio)
    pattern_n = int(pool_size * config.pool_pattern_ratio)
    random_n = pool_size - interesting_n - pattern_n

    if config.use_interesting_polynomials and interesting_entries:
        eligible = [
            entry
            for entry in interesting_entries
            if entry.get("shortest_length", 0) <= complexity
        ]
        if not eligible:
            eligible = interesting_entries
        random.shuffle(eligible)
        for entry in eligible:
            if len(pool) >= interesting_n:
                break
            add_target(entry["expr"], entry["encoding"].clone().unsqueeze(0))

    attempts = 0
    max_attempts = pool_size * 10
    while len(pool) < interesting_n + pattern_n and attempts < max_attempts:
        attempts += 1
        sampled_complexity = random.randint(1, max(1, complexity))
        actions, target_expr, encoding = generate_mixed_circuit(
            config, sampled_complexity, seen_polynomials=seen
        )
        if target_expr is None or encoding is None:
            continue
        add_target(target_expr, encoding)

    attempts = 0
    while len(pool) < pool_size and attempts < max_attempts:
        attempts += 1
        sampled_complexity = random.randint(1, max(1, complexity))
        actions_gen, polynomials_gen = generate_random_circuit(
            config.n_variables, sampled_complexity, mod=config.mod
        )
        if not polynomials_gen:
            continue
        target_expr = polynomials_gen[-1]
        encoding = encode_actions_with_compact_encoder(actions_gen, config).unsqueeze(0)
        add_target(target_expr, encoding)

    if not pool:
        sampled_complexity = random.randint(1, max(1, complexity))
        actions_gen, polynomials_gen = generate_random_circuit(
            config.n_variables, sampled_complexity, mod=config.mod
        )
        if polynomials_gen:
            pool.append(
                (
                    polynomials_gen[-1],
                    encode_actions_with_compact_encoder(actions_gen, config).unsqueeze(0),
                )
            )
    return pool


def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor):
    masked_logits = logits.masked_fill(~mask, -1e9)
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    log_probs = log_probs.masked_fill(~mask, 0.0)
    probs = torch.exp(log_probs) * mask.float()
    return log_probs, probs


def sample_action_from_policy(model, state, temperature: float = 1.0):
    circuit_graph, target_poly, mask = state
    batched_graph = Batch.from_data_list([circuit_graph.to(device)])
    target_poly = target_poly.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _, _ = model(batched_graph, target_poly, mask)
        masked_logits = logits / max(temperature, 1e-6)
        dist = Categorical(logits=masked_logits)
        action = dist.sample().item()
    return action


def build_game_from_target(target_poly_expr, target_encoding, config: Config):
    return Game(target_poly_expr, target_encoding, config).to("cpu")


def collect_experience(
    model,
    config: Config,
    planner: Optional[MCTSPlanner],
    buffer: ReplayBuffer,
    current_complexity: int,
    target_pool=None,
):
    model.eval()
    collected_steps = 0
    games_played = 0
    success_count = 0
    circuit_examples = []
    iteration_successes = []

    interesting_entries = load_interesting_circuit_data(config)
    seen_polynomials = set()
    seen_targets = set()

    def is_trivial_target(expr):
        return expr == sympy.Integer(1)

    with torch.no_grad():
        pbar = tqdm.tqdm(
            total=config.steps_per_iter,
            desc="Collecting SAC rollouts",
            disable=not config.show_progress_bars,
        )
        while collected_steps < config.steps_per_iter:
            games_played += 1
            target_poly_expr = None
            target_poly_encoding = None
            target_complexity = random.randint(1, max(1, current_complexity))
            for _ in range(50):
                if config.training_target_mode == "pool" and target_pool:
                    target_poly_expr, target_poly_encoding = random.choice(target_pool)
                elif config.training_target_mode == "dataset":
                    if interesting_entries:
                        eligible = [
                            entry
                            for entry in interesting_entries
                            if entry.get("shortest_length", 0) <= target_complexity
                        ]
                        if not eligible:
                            eligible = interesting_entries
                        entry = random.choice(eligible)
                        target_poly_expr = entry["expr"]
                        target_poly_encoding = entry["encoding"].clone().unsqueeze(0)
                elif config.training_target_mode == "mixed":
                    actions_gen, target_poly_expr, target_poly_encoding = generate_mixed_circuit(
                        config, target_complexity, seen_polynomials=seen_polynomials
                    )
                else:
                    actions_gen, polynomials_gen = generate_random_circuit(
                        config.n_variables, target_complexity, mod=config.mod
                    )
                    if not polynomials_gen:
                        continue
                    target_poly_expr = polynomials_gen[-1]
                    target_poly_encoding = encode_actions_with_compact_encoder(
                        actions_gen, config
                    ).unsqueeze(0)

                if target_poly_expr is None or target_poly_encoding is None:
                    continue
                if is_trivial_target(target_poly_expr):
                    continue
                target_key = str(sympy.expand(target_poly_expr))
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)
                break

            if target_poly_expr is None or target_poly_encoding is None:
                continue

            game = build_game_from_target(
                target_poly_expr, target_poly_encoding, config
            )

            reward = 0.0
            while not game.is_done() and collected_steps < config.steps_per_iter:
                state_tuple = game.observe()
                state = extract_state(state_tuple)

                mcts_pi = torch.zeros(config.max_actions, dtype=torch.float)
                has_mcts = False
                action = None
                if planner is not None and random.random() < config.mcts_policy_mix:
                    action, policy_dict = planner.select_action_with_policy(
                        game, temperature=config.mcts_policy_temperature
                    )
                    if action is not None and policy_dict:
                        mcts_pi = policy_dict_to_vector(
                            policy_dict, config.max_actions
                        )
                        has_mcts = True

                if action is None:
                    action = sample_action_from_policy(
                        model, state, temperature=config.action_temperature
                    )

                game.take_action(action)
                rewards = game.compute_rewards()
                reward = rewards[-1] if rewards else 0.0
                done = game.is_done()
                next_state_tuple = game.observe()
                next_state = extract_state(next_state_tuple)

                buffer.add(
                    state, action, reward, next_state, float(done), mcts_pi, has_mcts
                )
                collected_steps += 1
                pbar.update(1)

            success = game.is_success()
            if success:
                success_count += 1
            iteration_successes.append(success)
            if len(circuit_examples) < 5:
                circuit_examples.append(
                    {
                        "target": target_poly_expr,
                        "success": success,
                        "reward": reward,
                        "steps": len(game.actions),
                    }
                )

        pbar.close()

    success_rate = 100 * success_count / max(1, games_played)
    return success_rate, circuit_examples, iteration_successes


def soft_update(target_net, source_net, tau: float):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def sac_update(model, target_model, optimizer, buffer: ReplayBuffer, config: Config):
    if len(buffer) < config.min_buffer_size:
        return {}

    model.train()
    metrics = {
        "q_loss": 0.0,
        "policy_loss": 0.0,
        "ce_loss": 0.0,
        "updates": 0,
    }

    for _ in range(config.updates_per_iter):
        batch = buffer.sample(config.batch_size)
        states, actions, rewards, next_states, dones, mcts_pis, has_mcts = zip(*batch)

        graphs = [state[0].to("cpu") for state in states]
        targets = torch.stack([state[1] for state in states]).to(device)
        masks = torch.stack([state[2] for state in states]).to(device)

        next_graphs = [state[0].to("cpu") for state in next_states]
        next_targets = torch.stack([state[1] for state in next_states]).to(device)
        next_masks = torch.stack([state[2] for state in next_states]).to(device)

        actions_t = torch.tensor(actions, device=device, dtype=torch.long)
        rewards_t = torch.tensor(rewards, device=device, dtype=torch.float)
        dones_t = torch.tensor(dones, device=device, dtype=torch.float)

        mcts_pi_t = torch.stack(list(mcts_pis)).to(device)
        mcts_mask_t = torch.tensor(has_mcts, device=device, dtype=torch.bool)

        batched_graphs = Batch.from_data_list(graphs).to(device)
        batched_next_graphs = Batch.from_data_list(next_graphs).to(device)

        action_logits, q1, q2 = model(batched_graphs, targets, masks)
        q1_a = q1.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q2_a = q2.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_action_logits, _, _ = model(
                batched_next_graphs, next_targets, next_masks
            )
            next_log_probs, next_probs = masked_log_probs(
                next_action_logits, next_masks
            )
            _, target_q1, target_q2 = target_model(
                batched_next_graphs, next_targets, next_masks
            )
            min_target_q = torch.min(target_q1, target_q2)
            min_target_q = min_target_q.masked_fill(~next_masks, 0.0)
            next_v = (next_probs * (min_target_q - config.alpha * next_log_probs)).sum(
                dim=1
            )
            target_q = rewards_t + (1.0 - dones_t) * config.gamma * next_v

        q1_loss = F.mse_loss(q1_a, target_q)
        q2_loss = F.mse_loss(q2_a, target_q)
        q_loss = q1_loss + q2_loss

        log_probs, probs = masked_log_probs(action_logits, masks)
        min_q = torch.min(q1, q2).masked_fill(~masks, 0.0)
        policy_loss = (probs * (config.alpha * log_probs - min_q)).sum(dim=1).mean()

        ce_loss = torch.tensor(0.0, device=device)
        if config.mcts_ce_coef > 0.0 and mcts_mask_t.any():
            selected_pi = mcts_pi_t[mcts_mask_t]
            selected_mask = masks[mcts_mask_t].float()
            selected_pi = selected_pi * selected_mask
            selected_pi = selected_pi / (selected_pi.sum(dim=1, keepdim=True) + 1e-8)
            selected_log_probs = log_probs[mcts_mask_t]
            ce_loss = -(selected_pi * selected_log_probs).sum(dim=1).mean()
            policy_loss = policy_loss + config.mcts_ce_coef * ce_loss

        loss = q_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        soft_update(target_model, model, config.tau)

        metrics["q_loss"] += q_loss.item()
        metrics["policy_loss"] += policy_loss.item()
        metrics["ce_loss"] += ce_loss.item() if ce_loss is not None else 0.0
        metrics["updates"] += 1

    if metrics["updates"] > 0:
        for key in ("q_loss", "policy_loss", "ce_loss"):
            metrics[key] = metrics[key] / metrics["updates"]
    return metrics


def main():
    config = Config()
    print(f"Compact encoder size: {config.compact_size}")

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "n_variables": config.n_variables,
                "max_complexity": config.max_complexity,
                "max_degree": config.max_degree,
                "hidden_dim": config.hidden_dim,
                "embedding_dim": config.embedding_dim,
                "num_gnn_layers": config.num_gnn_layers,
                "num_transformer_layers": config.num_transformer_layers,
                "transformer_heads": config.transformer_heads,
                "transformer_dropout": config.transformer_dropout,
                "mod": config.mod,
                "gamma": config.gamma,
                "alpha": config.alpha,
                "tau": config.tau,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "buffer_size": config.buffer_size,
                "min_buffer_size": config.min_buffer_size,
                "steps_per_iter": config.steps_per_iter,
                "updates_per_iter": config.updates_per_iter,
                "action_temperature": config.action_temperature,
                "step_penalty": config.step_penalty,
                "success_reward": config.success_reward,
                "use_mcts": config.use_mcts,
                "mcts_simulations": config.mcts_simulations,
                "mcts_exploration": config.mcts_exploration,
                "mcts_policy_mix": config.mcts_policy_mix,
                "mcts_policy_temperature": config.mcts_policy_temperature,
                "mcts_ce_coef": config.mcts_ce_coef,
                "complexity_threshold": config.complexity_threshold,
                "complexity_window": config.complexity_window,
                "training_target_mode": config.training_target_mode,
                "use_synthetic_dataset": config.use_synthetic_dataset,
                "synthetic_samples": config.synthetic_samples,
                "synthetic_complexity_min": config.synthetic_complexity_min,
                "synthetic_complexity_max": config.synthetic_complexity_max,
            },
        )

    model = SACCircuitBuilder(config, config.compact_size).to(device)
    target_model = SACCircuitBuilder(config, config.compact_size).to(device)
    target_model.load_state_dict(model.state_dict())

    if config.resume and os.path.exists(config.resume_path):
        print(f"Loaded checkpoint: {config.resume_path}")
        model.load_state_dict(torch.load(config.resume_path, map_location=device))
        target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.rl_eps)

    buffer = ReplayBuffer(config.buffer_size)
    planner = MCTSPlanner(config) if config.use_mcts else None

    if config.use_synthetic_dataset:
        synthetic_transitions = generate_synthetic_transitions(
            config,
            samples_n=config.synthetic_samples,
            complexity_min=config.synthetic_complexity_min,
            complexity_max=config.synthetic_complexity_max,
        )
        for transition in synthetic_transitions:
            buffer.add(*transition)
        print(f"Prefilled replay buffer with {len(synthetic_transitions)} synthetic steps")

    current_complexity = 1
    recent_successes = []

    target_pool_by_complexity = {}

    last_iteration = 0
    last_success_rate = 0.0
    last_metrics = {}
    best_success_rate = -1.0
    best_iteration = 0
    interrupted = False

    try:
        for iteration in range(1, 10001):
            if config.training_target_mode == "pool":
                pool = target_pool_by_complexity.get(current_complexity)
                if pool is None:
                    pool = build_target_pool(
                        config,
                        current_complexity,
                        load_interesting_circuit_data(config),
                    )
                    target_pool_by_complexity[current_complexity] = pool
                target_pool = pool
            else:
                target_pool = None

            success_rate, circuit_examples, iteration_successes = collect_experience(
                model, config, planner, buffer, current_complexity, target_pool=target_pool
            )

            last_iteration = iteration
            last_success_rate = success_rate
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_iteration = iteration

            recent_successes.extend(iteration_successes)
            recent_successes = recent_successes[-config.complexity_window:]
            if len(recent_successes) >= config.complexity_window // 2:
                recent_success_rate = sum(recent_successes) / len(recent_successes)
                if (
                    config.allow_complexity_backoff
                    and recent_success_rate < config.sr_backoff_threshold
                    and current_complexity > 1
                ):
                    current_complexity -= 1
                    recent_successes = []
                    print(
                        f"*** Complexity Decreased to {current_complexity} (SR: {recent_success_rate:.2f}) ***"
                    )
                elif (
                    recent_success_rate >= config.sr_advance_threshold
                    and current_complexity < config.max_complexity
                ):
                    current_complexity += 1
                    recent_successes = []
                    print(
                        f"*** Complexity Increased to {current_complexity} (SR: {recent_success_rate:.2f}) ***"
                    )

            metrics = sac_update(model, target_model, optimizer, buffer, config)
            last_metrics = metrics
            print(
                f"Iter {iteration}: SR {success_rate:.1f}%, "
                f"Q {metrics.get('q_loss', 0.0):.4f}, "
                f"Pi {metrics.get('policy_loss', 0.0):.4f}, "
                f"CE {metrics.get('ce_loss', 0.0):.4f}, "
                f"Buffer {len(buffer)}"
            )
            if config.use_wandb:
                wandb.log(
                    {
                        "success_rate": success_rate,
                        "q_loss": metrics.get("q_loss", 0.0),
                        "policy_loss": metrics.get("policy_loss", 0.0),
                        "ce_loss": metrics.get("ce_loss", 0.0),
                        "buffer_size": len(buffer),
                        "complexity": current_complexity,
                    },
                    step=iteration,
                )
            for i, ex in enumerate(circuit_examples):
                print(
                    f"  Ex {i + 1}: {'Success' if ex['success'] else 'Fail'} "
                    f"(R: {ex['reward']:.2f}, Steps: {ex['steps']}) Target: {ex['target']}"
                )

            if iteration % 50 == 0:
                path = f"sac_model_{config.model_tag}_n{config.n_variables}_C{config.max_complexity}.pt"
                torch.save(model.state_dict(), path)
                print(f"  Model saved to {path}")
    except KeyboardInterrupt:
        interrupted = True
        interrupt_path = (
            f"sac_model_{config.model_tag}_n{config.n_variables}_C{config.max_complexity}_interrupt.pt"
        )
        torch.save(model.state_dict(), interrupt_path)
        print(f"\nTraining interrupted. Model saved to {interrupt_path}")

    if last_iteration == 0:
        print("\n=== Training Summary ===")
        print("No iterations completed.")
        if interrupted:
            print("Run status: interrupted")
        return

    print("\n=== Training Summary ===")
    print(f"Total iterations: {last_iteration}")
    print(f"Final complexity: {current_complexity}")
    if best_success_rate >= 0:
        print(f"Best SR: {best_success_rate:.1f}% (Iter {best_iteration})")
    print(f"Final SR: {last_success_rate:.1f}%")
    print(
        "Final losses: "
        f"Q {last_metrics.get('q_loss', 0.0):.4f}, "
        f"Pi {last_metrics.get('policy_loss', 0.0):.4f}, "
        f"CE {last_metrics.get('ce_loss', 0.0):.4f}"
    )
    print(f"Buffer size: {len(buffer)}")
    if interrupted:
        print("Run status: interrupted")


if __name__ == "__main__":
    main()
