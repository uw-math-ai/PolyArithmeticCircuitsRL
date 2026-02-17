import streamlit as st

import sys
import os
import json
import tempfile
import copy
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import sympy as sp
from torch_geometric.data import Batch


# --- Constants ---
BEAM_WIDTH = 20  # Hardcoded as requested to prevent loop failures


# --- Path Setup ---
current_dir = Path(__file__).resolve().parent
src_root = current_dir.parent
project_root = src_root.parent

for p in [current_dir, src_root, project_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# --- Imports ---
try:
    from PPO import CircuitBuilder, Config, encode_actions_with_compact_encoder
    from State import Game
    from generator import generate_random_circuit
    from utils import decode_action, encode_action
    from encoders.compact_encoder import CompactOneHotGraphEncoder
    from pyvis.network import Network
except ImportError as e:
    st.error(f"Import Error: {e}. Make sure you are running this from the correct directory.")
    st.stop()


# --- Helper Functions ---

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sympy_to_vector_encoding(poly, n_vars, max_complexity, mod):
    """
    Converts a SymPy polynomial into a dense vector (Monomial Encoding).
    Uses 'Reverse' order (x_{n-1} is Least Significant) to match standard iteration.
    """
    d = max_complexity * 2
    base = d + 1
    vector_size = base ** n_vars

    vec = torch.zeros(vector_size, dtype=torch.float)

    if poly is None:
        return vec

    poly = sp.expand(poly)
    if poly.is_Number:
        vec[0] = float(poly) % mod
        return vec

    try:
        terms_dict = poly.as_coefficients_dict()
    except Exception:
        return vec

    symbols = sorted([sp.Symbol(f"x{i}") for i in range(n_vars)], key=lambda s: s.name)

    for term, coeff in terms_dict.items():
        index = 0
        valid_term = True

        for i, sym in enumerate(symbols):
            degree = sp.degree(term, gen=sym)
            if degree > d:
                valid_term = False
                break
            weight = base ** (n_vars - 1 - i)
            index += degree * weight

        if valid_term and index < vector_size:
            vec[index] = float(coeff) % mod

    return vec


@st.cache_resource(show_spinner=False)
def load_model_cached(n_vars: int, max_complexity: int, modulus: int, model_filename: str):
    config = Config()
    config.n_variables = int(n_vars)
    config.max_complexity = int(max_complexity)
    config.mod = int(modulus)

    config.compact_size = CompactOneHotGraphEncoder(
        N=config.max_complexity, P=config.mod, D=config.n_variables
    ).size

    device = get_device()

    paths = [
        model_filename,
        os.path.join(current_dir, model_filename),
        os.path.join(current_dir, "Trained Model", model_filename),
        os.path.join("Trained Model", model_filename),
    ]

    full_path = next((p for p in paths if os.path.exists(p)), None)

    ckpt_loaded = None
    ckpt_error = None
    model_mode = "graph"

    if full_path:
        try:
            # 1. Try Graph Model
            model = CircuitBuilder(config, config.compact_size).to(device)
            state = torch.load(full_path, map_location=device)
            model.load_state_dict(state, strict=False)
            model.eval()
            ckpt_loaded = str(full_path)

        except RuntimeError as e:
            # 2. Try Vector Model (1331 size)
            msg = str(e)
            if "size mismatch" in msg and "1331" in msg:
                d = config.max_complexity * 2
                vector_size = (d + 1) ** config.n_variables
                config.compact_size = vector_size

                model = CircuitBuilder(config, config.compact_size).to(device)

                try:
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    ckpt_loaded = str(full_path)
                    model_mode = "vector"
                except Exception as e2:
                    ckpt_error = f"Retry failed: {e2}"
            else:
                ckpt_error = f"{os.path.basename(full_path)}: {e}"
    else:
        model = CircuitBuilder(config, config.compact_size).to(device)
        ckpt_error = f"File not found: {model_filename}"

    return {
        "model": model,
        "config": config,
        "device": device,
        "ckpt": ckpt_loaded,
        "ckpt_error": ckpt_error,
        "mode": model_mode,
    }


def _subscript_number(num: int) -> str:
    return str(num).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _variable_label(idx: int) -> str:
    return f"𝑥{_subscript_number(idx)}"


def build_circuit_network(actions, polynomials, n_vars, theme):
    # Professional Color Palette
    if theme == "Dark":
        bg_color = "#1f2937"
        text_color = "#f9fafb"
        node_input = "#374151" 
        node_const = "#4b5563"
        node_op = "#10b981" # Emerald Green
        edge_color = "#6b7280"
    else:
        bg_color = "#ffffff"
        text_color = "#111827"
        node_input = "#f3f4f6" # Light Gray
        node_const = "#f9fafb"
        node_op = "#dbeafe" # Light Blue
        edge_color = "#9ca3af"

    net = Network(height="600px", width="100%", bgcolor=bg_color, font_color=text_color, directed=True)
    
    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "DU",
                "sortMethod": "directed",
                "levelSeparation": 120,
                "nodeSpacing": 180,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True
            }
        },
        "physics": {"enabled": False},
        "edges": {
            "smooth": {"type": "cubicBezier", "roundness": 0.4},
            "color": {"color": edge_color, "highlight": "#2563eb"},
            "width": 2,
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}
        },
        "nodes": {
            "borderWidth": 1,
            "borderWidthSelected": 2,
            "shape": "box",
            "font": {"face": "Inter", "size": 16, "color": text_color},
            "shadow": {"enabled": True, "color": "rgba(0,0,0,0.1)", "size": 10, "x": 0, "y": 4}
        },
        "interaction": {"hover": True, "zoomView": False}
    }
    net.set_options(json.dumps(options))

    # Base Nodes
    for i in range(n_vars):
        net.add_node(i, label=f"x{i}", shape="circle", color=node_input, size=25, level=0)
    net.add_node(n_vars, label="1", shape="circle", color=node_const, size=25, level=0)

    levels = {i: 0 for i in range(n_vars + 1)}

    for i, action in enumerate(actions):
        op, n1, n2 = action
        if op in ["input", "constant"]: continue
        
        lvl1 = levels.get(n1, 0)
        lvl2 = levels.get(n2, 0)
        curr_level = max(lvl1, lvl2) + 1
        levels[i] = curr_level
        
        poly_str = str(polynomials[i]) if i < len(polynomials) else "?"
        symbol = "+" if op == "add" else "×"
        color = "#e0f2fe" if op == "add" else "#dcfce7" # Subtle blue vs green
        
        # Tooltip for interaction
        title_text = f"Operation: {op.upper()}\nInputs: Node {n1}, Node {n2}\nResult: {poly_str}"
        
        net.add_node(
            i, 
            label=symbol, 
            title=title_text, 
            shape="box", 
            color=color, 
            level=curr_level,
            margin=12
        )
        if n1 != -1: net.add_edge(n1, i)
        if n2 != -1: net.add_edge(n2, i)

    return net


def perform_beam_search(model, game, beam_width, max_depth, device):
    candidates = [(game, 0.0, [])]
    finished_candidates = []

    for depth in range(max_depth):
        next_candidates = []

        for curr_game, curr_score, curr_steps in candidates:
            if curr_game.is_done():
                finished_candidates.append((curr_game, curr_score, curr_steps))
                continue

            state = curr_game.observe()
            circuit_graph, target_t, circuit_acts, mask = state

            # Ensure mask is on device and 2D
            mask = mask.to(device)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

            with torch.no_grad():
                batched_graph = Batch.from_data_list([circuit_graph.to(device)])

                # Model forward
                logits, value = model(
                    batched_graph,
                    target_t.to(device).unsqueeze(0),
                    circuit_acts,
                    mask,  # No extra unsqueeze!
                )

            # Hard masking to prevent invalid actions
            logits = logits[0].masked_fill(~mask[0], float("-inf"))
            probs = F.softmax(logits, dim=0)

            # Get valid top k
            valid_count = mask[0].sum().item()
            k = min(beam_width, valid_count, len(probs))
            if k == 0:
                continue

            top_probs, top_indices = torch.topk(probs, k=k)

            for prob, action_idx in zip(top_probs, top_indices):
                if prob == 0 or action_idx.item() >= mask.size(1) or not mask[0][action_idx]:
                    continue

                # Use deepcopy to avoid shared mutable state across beam branches
                new_game = copy.deepcopy(curr_game)
                new_game.take_action(action_idx.item())

                op, n1, n2 = decode_action(action_idx.item(), new_game.max_nodes)
                step_info = {
                    "operation": op,
                    "node1": n1,
                    "node2": n2,
                    "result": str(new_game.polynomials[-1]),
                    "prob": prob.item(),
                }

                next_candidates.append(
                    (new_game, curr_score + math.log(prob.item() + 1e-9), curr_steps + [step_info])
                )

        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = next_candidates[:beam_width]

        if not candidates:
            break

        finished_candidates.extend(candidates)

    def is_success(g):
        return g.polynomials and sp.expand(g.polynomials[-1] - g.target_poly_expr) == 0

    finished_candidates.sort(
        key=lambda x: (is_success(x[0]), x[1]),
        reverse=True,
    )

    if not finished_candidates:
        return [], False, game

    return finished_candidates[0][2], is_success(finished_candidates[0][0]), finished_candidates[0][0]


# --- Main App ---

def main():
    st.set_page_config(page_title="Arithmetic Circuit Visualizer", layout="wide")
    st.title("Arithmetic Circuit Visualizer (PPO/Hybrid)")

    with st.sidebar:
        st.header("Controls")
        theme_choice = st.radio("Theme", ["Light", "Dark"], horizontal=True)
        n_vars = st.number_input("Variables (N)", 1, 6, 3)
        max_comp = st.number_input("Complexity (C)", 1, 10, 5)
        # Modulus removed from UI for now
        model_file = st.text_input("Model File", "polynomial_net_complexity.pth")

        st.divider()
        max_depth = st.slider("Max Steps", 1, 20, 10)

    # Use a fixed modulus internally (same default as before)
    FIXED_MODULUS = 50
    cache = load_model_cached(n_vars, max_comp, FIXED_MODULUS, model_file)

    if cache["ckpt"]:
        st.success(f"Loaded: {os.path.basename(cache['ckpt'])}")
        st.caption(f"Mode: {cache['mode']}")
    else:
        st.warning(f"Error: {cache['ckpt_error']}")

    if "target_poly" not in st.session_state:
        st.session_state["target_poly"] = None
    if "target_actions" not in st.session_state:
        st.session_state["target_actions"] = None
    if "result" not in st.session_state:
        st.session_state["result"] = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate Random Polynomial", type="primary"):
            actions, polys = generate_random_circuit(
                cache["config"].n_variables,
                cache["config"].max_complexity,
                mod=cache["config"].mod,
            )
            st.session_state["target_poly"] = polys[-1]
            st.session_state["target_actions"] = actions
            st.session_state["result"] = None

    with col2:
        txt_in = st.text_input("Manual Polynomial (e.g. x0 + x1)")
        if st.button("Use Entered Polynomial"):
            try:
                vars_str = [f"x{i}" for i in range(n_vars)]
                syms = sp.symbols(vars_str)
                local_d = {f"x{i}": s for i, s in enumerate(syms)}
                p = sp.expand(sp.sympify(txt_in, locals=local_d))

                st.session_state["target_poly"] = p
                st.session_state["target_actions"] = None
                st.session_state["result"] = None
            except Exception as e:
                st.error(f"Parse error: {e}")

    if st.session_state["target_poly"] is not None:
        st.subheader("Target Polynomial")
        st.latex(sp.latex(st.session_state["target_poly"]))

    if st.button("Solve Circuit"):
        with st.spinner("Running Beam Search..."):
            config = cache["config"]
            model = cache["model"]
            device = cache["device"]

            if cache["mode"] == "vector":
                target_enc = sympy_to_vector_encoding(
                    st.session_state["target_poly"],
                    config.n_variables,
                    config.max_complexity,
                    config.mod,
                )
            else:
                if st.session_state["target_actions"]:
                    target_enc = encode_actions_with_compact_encoder(
                        st.session_state["target_actions"],
                        config,
                    )
                else:
                    st.info("⚠️ Searching for a circuit that generates this polynomial...")
                    from find_circuit_for_polynomial import find_simple_circuit_for_polynomial

                    found_actions = find_simple_circuit_for_polynomial(
                        st.session_state["target_poly"],
                        config,
                        max_attempts=500,
                    )

                    if found_actions:
                        st.success("✓ Found a circuit! Encoding it properly.")
                        target_enc = encode_actions_with_compact_encoder(
                            found_actions,
                            config,
                        )
                        st.session_state["target_actions"] = found_actions
                    else:
                        st.warning(
                            "⚠️ Could not find circuit. Using empty encoding (model may struggle)."
                        )
                        enc = CompactOneHotGraphEncoder(
                            config.max_complexity, config.mod, config.n_variables
                        )
                        target_enc = torch.from_numpy(enc.get_encoding()).float()

            game = Game(st.session_state["target_poly"], target_enc, config).to(device)

            steps, success, final_game = perform_beam_search(
                model,
                game,
                BEAM_WIDTH,
                max_depth,
                device,
            )

            st.session_state["result"] = {
                "steps": steps,
                "success": success,
                "game_actions": final_game.actions,
                "game_polys": final_game.polynomials,
            }

    if st.session_state["result"]:
        res = st.session_state["result"]

        st.divider()

        if res["success"]:
            st.success("Target Reached!")
        else:
            st.warning("Target Not Reached. Showing best path.")

        theme_palette = {
            "Light": {
                "bg": "#ffffff",
                "text": "#000000",
                "input_node": "#97c2fc",
                "const_node": "#ffff00",
                "op_node": "#fb7e81",
                "edge": "#666666",
            },
            "Dark": {
                "bg": "#222222",
                "text": "#ffffff",
                "input_node": "#1b263b",
                "const_node": "#415a77",
                "op_node": "#778da9",
                "edge": "#e0e1dd",
            },
        }

        col_graph, col_list = st.columns([2, 1])

        with col_graph:
            net = build_circuit_network(
                res["game_actions"],
                res["game_polys"],
                n_vars,
                theme_palette[theme_choice],
            )

            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tf:
                net.save_graph(tf.name)
                html_data = Path(tf.name).read_text()

            st.components.v1.html(html_data, height=750)

        with col_list:
            for i, step in enumerate(res["steps"]):
                st.markdown(f"**{i+1}. {step['operation']}** ({step['node1']}, {step['node2']})")
                st.latex(f"\\rightarrow {sp.latex(sp.sympify(step['result']))}")
                st.caption(f"Prob: {step['prob']:.4f}")


if __name__ == "__main__":
    main()
