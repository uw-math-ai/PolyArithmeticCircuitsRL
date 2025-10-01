import os
import sys
from pathlib import Path
import json

import streamlit as st
from pyvis.network import Network

import torch
import sympy as sp
import tempfile

# Make repo root and transformer folder importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TRANSFORMER_DIR = REPO_ROOT / "transformer"
if str(TRANSFORMER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFORMER_DIR))

# Import project modules
from transformer.fourthGen import CircuitBuilder, Config
from transformer.generator import (
    generate_monomials_with_additive_indices,
    generate_random_circuit,
)
from transformer.utils import vector_to_sympy
from transformer.State import Game
from transformer.test_model import (
    sympy_to_vector,
    hybrid_tree_search_top_w,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource(show_spinner=False)
def load_model_and_index(n_vars: int, max_complexity: int, modulus: int):
    config = Config()
    config.n_variables = int(n_vars)
    config.max_complexity = int(max_complexity)
    config.mod = int(modulus)

    n = config.n_variables
    d = config.max_complexity * 2
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(n, d)

    # Calculate max_vector_size (dense vector length)
    base = d + 1
    max_idx = 0
    for i in range(n):
        max_idx += d * (base ** i)
    max_vector_size = max_idx + 1

    device = get_device()
    model = CircuitBuilder(config, max_vector_size).to(device)

    # Try to load PPO checkpoint; fallback to best supervised
    ckpt_paths = [
        REPO_ROOT / f"transformer/ppo_model_n{config.n_variables}_C{config.max_complexity}_curriculum.pt",
        REPO_ROOT / f"transformer/best_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt",
    ]

    ckpt_loaded = None
    ckpt_error = None
    for path in ckpt_paths:
        if path.exists():
            try:
                state = torch.load(path, map_location=device)
                model.load_state_dict(state, strict=False)
                ckpt_loaded = str(path)
                ckpt_error = None
                break
            except Exception as err:  # pragma: no cover - runtime safeguard
                ckpt_error = f"{path.name}: {err}"
                continue

    return {
        "model": model,
        "config": config,
        "index_to_monomial": index_to_monomial,
        "monomial_to_index": monomial_to_index,
        "max_vector_size": max_vector_size,
        "device": device,
        "ckpt": ckpt_loaded,
        "ckpt_error": ckpt_error,
    }


def _subscript_number(num: int) -> str:
    subs = str(num).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))
    return subs


def _variable_label(idx: int) -> str:
    italic_x = "𝑥"  # Mathematical italic small x (U+1D465)
    return f"{italic_x}{_subscript_number(idx)}"


def build_circuit_network(steps, n_vars, theme):
    """Create a modern, inverted (inputs at bottom) PyVis network for the circuit."""
    net = Network(height="720px", width="100%", bgcolor=theme["bg"], font_color=theme["text"], directed=True)

    options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "DU",  # Draw from bottom (inputs) upward
                "sortMethod": "directed",
                "levelSeparation": 110,
                "nodeSpacing": 220,
                "treeSpacing": 260,
            }
        },
        "interaction": {
            "hover": True,
            "zoomView": False,
            "dragView": True,
        },
        "physics": {"enabled": False},
        "edges": {
            "smooth": {"type": "cubicBezier", "roundness": 0.25},
            "color": theme["edge"],
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.42}},
            "width": 3
        },
        "nodes": {
            "shadow": {
                "enabled": True,
                "size": 12,
                "x": 0,
                "y": 4
            },
            "borderWidth": 3,
        },
    }
    net.set_options(json.dumps(options))

    # Common font style
    font_common = {
        "size": 56,
        "face": "Times New Roman, Latin Modern Math, Cambria Math, serif",
        "bold": False,
        "color": theme["text"],
    }

    levels = {}

    # Base nodes: x0..x{n-1}, const 1 at index n (placed at the bottom via hierarchical 'DU')
    for i in range(n_vars):
        label = _variable_label(i)
        net.add_node(
            i,
            label=label,
            shape="circle",
            color=theme["input_node"],
            borderWidth=3,
            font=font_common,
            size=70,
            level=0,
            title=f"$x_{{{i}}}$",
            margin=20,
        )
        levels[i] = 0

    const_idx = n_vars
    net.add_node(
        const_idx,
        label="1",
        shape="circle",
        color=theme["const_node"],
        borderWidth=3,
        font=font_common,
        size=70,
        level=0,
        title="$1$",
        margin=20,
    )
    levels[const_idx] = 0

    # Operation nodes start from n_vars + 1
    for i, step in enumerate(steps):
        node_id = n_vars + 1 + i
        op = step.get("operation", "?")
        symbol = "＋" if op == "add" else "×" if op == "multiply" else op
        title = str(step.get("result", ""))
        n1, n2 = step.get("node1"), step.get("node2")
        parent_levels = [levels.get(idx, 0) for idx in (n1, n2) if idx is not None]
        node_level = (max(parent_levels) if parent_levels else 0) + 1

        net.add_node(
            node_id,
            label=symbol,
            title=title,
            shape="circle",
            color=theme["op_node"],
            borderWidth=3,
            font={**font_common, "size": 88},
            size=110,
            level=node_level,
            margin=24,
        )
        levels[node_id] = node_level

        if n1 is not None:
            net.add_edge(n1, node_id, color=theme["edge"], width=3)
        if n2 is not None:
            net.add_edge(n2, node_id, color=theme["edge"], width=3)
    return net


def generate_random_target(index_to_monomial, monomial_to_index, n, d, C, mod):
    actions, polys, _, _ = generate_random_circuit(n, d, C, mod=mod)
    vec = polys[-1]
    poly_sp = vector_to_sympy(vec, index_to_monomial)
    return poly_sp


def main():
    st.set_page_config(page_title="Arithmetic Circuit Visualizer", layout="wide")
    st.title("Arithmetic Circuit Visualizer")
    default_config = Config()

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        theme_choice = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)

        st.subheader("Problem settings")
        n_vars_input = st.number_input(
            "Variables (N)",
            min_value=1,
            max_value=6,
            value=int(default_config.n_variables),
            step=1,
        )
        max_complexity_input = st.number_input(
            "Max complexity (C)",
            min_value=1,
            max_value=12,
            value=int(default_config.max_complexity),
            step=1,
        )
        modulus_input = st.number_input(
            "Coefficient modulus",
            min_value=2,
            max_value=200,
            value=int(default_config.mod),
            step=1,
        )
        st.caption("Settings control polynomial generation and which checkpoint is loaded.")

        st.divider()
        st.subheader("Search parameters")
        beam_w = st.number_input("Top-w (tree search)", min_value=2, max_value=50, value=10, step=1)
        depth_d = st.number_input("Depth (tree search)", min_value=1, max_value=50, value=15, step=1)
        st.caption("Higher values explore more but take longer.")

    n_vars = int(n_vars_input)
    max_complexity = int(max_complexity_input)
    modulus = int(modulus_input)

    cache = load_model_and_index(n_vars, max_complexity, modulus)
    model = cache["model"]
    config = cache["config"]
    index_to_monomial = cache["index_to_monomial"]
    monomial_to_index = cache["monomial_to_index"]
    max_vector_size = cache["max_vector_size"]
    device = cache["device"]

    if cache["ckpt"] is None:
        st.warning(
            "No trained checkpoint found for these settings. Model uses random initialization."
        )
        if cache.get("ckpt_error"):
            st.caption(cache["ckpt_error"])
    else:
        st.info(f"Loaded model: {cache['ckpt']}")

    if cache.get("ckpt_error") and cache["ckpt"] is None:
        st.warning(f"Checkpoint load error: {cache['ckpt_error']}")

    settings_key = (n_vars, max_complexity, modulus)
    if st.session_state.get("_settings_key") != settings_key:
        st.session_state["_settings_key"] = settings_key
        st.session_state["target_poly"] = None
        st.session_state["result"] = None

    # Session state for current target
    if "target_poly" not in st.session_state:
        st.session_state["target_poly"] = None

    # Buttons row
    col1, col2 = st.columns([1, 1])
    if "result" not in st.session_state:
        st.session_state["result"] = None

    with col1:
        if st.button("Generate Random Polynomial"):
            n = config.n_variables
            d = config.max_complexity * 2
            C = config.max_complexity
            target_sp = generate_random_target(index_to_monomial, monomial_to_index, n, d, C, config.mod)
            st.session_state["target_poly"] = sp.expand(target_sp)
            st.session_state["result"] = None

    with col2:
        var_list = ", ".join(f"x{i}" for i in range(config.n_variables))
        target_input = st.text_input(
            f"Enter polynomial (variables: {var_list})",
            value="",
            placeholder="e.g., x0*(x1 + 1) or x0**2 + 2*x0*x1 + x1**2",
        )
        if st.button("Use Entered Polynomial") and target_input.strip():
            try:
                vars_str = [f"x{i}" for i in range(config.n_variables)]
                symbols_tuple = sp.symbols(vars_str)
                local_dict = {name: symbol for name, symbol in zip(vars_str, symbols_tuple)}
                p = sp.expand(sp.sympify(target_input, locals=local_dict))
                st.session_state["target_poly"] = p
                st.session_state["result"] = None
            except Exception as e:
                st.error(f"Could not parse polynomial: {e}")

    # Theme palette and global styling
    theme_palette = {
        "Light": {
            "bg": "#f7f9fc",
            "text": "#101828",
            "input_node": "#d0e3ff",
            "const_node": "#ffe2a8",
            "op_node": "#c5f2d6",
            "edge": "#94a3b8",
            "page": "#ffffff",
            "sidebar": "#eef2fb",
            "button_bg": "#2563eb",
            "button_hover": "#1d4ed8",
            "button_text": "#f8fafc",
            "control_bg": "#ffffff",
            "border": "#c7d2fe",
            "placeholder": "#1f2a44",
        },
        "Dark": {
            "bg": "#0b0f17",
            "text": "#f1f5f9",
            "input_node": "#23344c",
            "const_node": "#59472a",
            "op_node": "#274131",
            "edge": "#64748b",
            "page": "#111827",
            "sidebar": "#141c2b",
            "button_bg": "#2563eb",
            "button_hover": "#1d4ed8",
            "button_text": "#f8fafc",
            "control_bg": "#1f2937",
            "border": "#334155",
            "placeholder": "#cbd5f5",
        },
    }
    theme = theme_palette[theme_choice]

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {theme['page']} !important;
            color: {theme['text']} !important;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        header[data-testid="stHeader"] {{
            background: linear-gradient(90deg, {theme['sidebar']}, {theme['page']});
            color: {theme['text']} !important;
        }}
        header[data-testid="stHeader"] * {{
            color: {theme['text']} !important;
        }}
        [data-testid="stToolbar"] {{
            background-color: {theme['sidebar']} !important;
        }}
        [data-testid="stToolbar"] * {{
            color: {theme['text']} !important;
        }}
        [data-testid="stSidebar"] {{
            background-color: {theme['sidebar']} !important;
            color: {theme['text']} !important;
        }}
        [data-testid="stSidebar"] * {{
            color: {theme['text']} !important;
        }}
        .stMarkdown, .stText, .stCaption, .stHeader, .stSubheader, .stRadio, .stNumberInput label {{
            color: {theme['text']} !important;
        }}
        .stButton>button {{
            background-color: {theme['button_bg']} !important;
            color: {theme['button_text']} !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.55rem 1.2rem !important;
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.25);
        }}
        .stButton>button:hover {{
            background-color: {theme['button_hover']} !important;
        }}
        input, textarea, select {{
            background-color: {theme['control_bg']} !important;
            color: {theme['text']} !important;
            border-radius: 8px !important;
            border: 1px solid {theme['border']} !important;
        }}
        [data-testid="stTextInput"] label {{
            color: {theme['text']} !important;
            font-weight: 600 !important;
        }}
        [data-testid="stTextInput"] div[role="tooltip"] {{
            color: {theme['text']} !important;
        }}
        [data-baseweb="input"] input {{
            background-color: {theme['control_bg']} !important;
            color: {theme['text']} !important;
            border-radius: 8px !important;
        }}
        [data-baseweb="input"] input::placeholder {{
            color: {theme['placeholder']} !important;
            opacity: 0.9 !important;
        }}
        textarea::placeholder {{
            color: {theme['placeholder']} !important;
            opacity: 0.9 !important;
        }}
        [data-baseweb="select"] > div {{
            background-color: {theme['control_bg']} !important;
            color: {theme['text']} !important;
        }}
        [data-testid="stAlert"] > div {{
            background-color: {theme['control_bg']} !important;
            color: {theme['text']} !important;
            border: 1px solid {theme['border']} !important;
            border-radius: 10px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display target and factorization
    target_sp = st.session_state.get("target_poly")
    if target_sp is not None:
        st.subheader("Target Polynomial")
        st.latex(sp.latex(target_sp))
        try:
            fact = sp.factor(target_sp)
            st.caption("Factorization")
            st.latex(sp.latex(fact))
        except Exception:
            st.caption("Factorization unavailable for this target.")

        evaluate_clicked = st.button("Evaluate Model and Build Circuit", type="primary")
        if evaluate_clicked:
            with st.spinner("Searching for circuit..."):
                # Convert target to vector
                n = config.n_variables
                poly_vec = sympy_to_vector(target_sp, monomial_to_index, n, config.mod, max_vector_size=max_vector_size).to(device)
                # Create game and search
                game = Game(target_sp, poly_vec, config, index_to_monomial, monomial_to_index).to(device)
                steps, success = hybrid_tree_search_top_w(config, model, game, w=int(beam_w), d=int(depth_d))
            target_key = sp.srepr(sp.expand(target_sp))
            st.session_state["result"] = {
                "steps": steps,
                "success": success,
                "target_key": target_key,
            }

        result = st.session_state.get("result")
        current_key = sp.srepr(sp.expand(target_sp))

        if result and result.get("target_key") == current_key:
            steps = result.get("steps", [])
            success = result.get("success", False)

            st.subheader("Search Result")
            if success:
                st.success("Successfully found an exact circuit.")
            else:
                st.warning("Exact circuit not found within limits. Showing best attempt.")

            if steps:
                st.markdown("Steps (latest is root):")
                for i, step in enumerate(steps):
                    op = step.get("operation", "")
                    n1, n2 = step.get("node1"), step.get("node2")
                    st.write(f"{i+1}. {op} (node {n1}, node {n2}) -> {step.get('result')}")

                if "tree_zoom_slider" not in st.session_state:
                    st.session_state["tree_zoom_slider"] = 1.0
                zoom_level = st.slider(
                    "Tree zoom",
                    min_value=0.6,
                    max_value=2.5,
                    value=float(st.session_state.get("tree_zoom_slider", 1.0)),
                    step=0.05,
                    key="tree_zoom_slider",
                    help="Adjust the visual scale of the circuit graph.",
                )
                zoom_val = float(zoom_level)

                net = build_circuit_network(steps, config.n_variables, theme)
                with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as tmp:
                    tmp_path = tmp.name
                net.save_graph(tmp_path)
                html = Path(tmp_path).read_text(encoding="utf-8")
                html = html.replace(
                    "network = new vis.Network(container, data, options);",
                    f"network = new vis.Network(container, data, options);\nnetwork.once('afterDrawing', function(){{ network.moveTo({{ scale: {zoom_val}, animation: {{ duration: 0 }} }}); }});",
                )
                st.components.v1.html(html, height=650, scrolling=True)
            else:
                st.info("No steps found.")

    else:
        st.info("Click 'Generate Random Polynomial' or enter one manually to begin.")


if __name__ == "__main__":
    main()
