"""
Streamlit-powered visual demo for arithmetic-circuit generation.

Run locally with:
    streamlit run demo_visualizer.py
"""
from __future__ import annotations

import random
import sympy
import streamlit as st

from transformer.generator import generate_random_circuit
from visualization.circuit_visualizer import render_circuit_html, summarize_actions


def build_circuit(n: int, C: int, mod: int, seed: int):
    rng = random.Random(seed)
    actions, polynomials = generate_random_circuit(n, C, mod=mod, rng=rng)
    return actions, polynomials


def main():
    st.set_page_config(page_title="Polynomial Circuit Demo", layout="wide")
    st.title("Polynomial Arithmetic Circuit Visualizer")
    st.write(
        "Generate a random arithmetic circuit, inspect each operation, and explore "
        "how the agent composes the final polynomial."
    )

    with st.sidebar:
        st.header("Circuit Settings")
        n = st.number_input("Number of variables (n)", min_value=1, max_value=8, value=3, step=1)
        C = st.number_input("Max operations (C)", min_value=1, max_value=15, value=5, step=1)
        mod = st.number_input("Modulus for coefficients", min_value=2, max_value=19, value=7, step=1)
        seed = st.number_input("Random seed", min_value=0, max_value=10000, value=0, step=1)
        regenerate = st.button("Generate circuit", type="primary")

    if "circuit_state" not in st.session_state or regenerate:
        actions, polynomials = build_circuit(n, C, mod, seed)
        st.session_state.circuit_state = {"actions": actions, "polynomials": polynomials}

    actions = st.session_state.circuit_state["actions"]
    polynomials = st.session_state.circuit_state["polynomials"]

    if not actions or not polynomials:
        st.warning("Could not generate a circuit with the current settings.")
        return

    final_poly = polynomials[-1]

    st.subheader("Target Polynomial")
    with st.expander("Expanded expression", expanded=True):
        st.latex(sympy.latex(sympy.expand(final_poly)))

    graph_html = render_circuit_html(actions, polynomials, height="720px")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Step-by-step actions")
        step_rows = summarize_actions(actions, polynomials)
        st.dataframe(step_rows, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Circuit graph")
        st.components.v1.html(graph_html, height=760, scrolling=True)

    st.caption(
        "Each node represents an intermediate polynomial. Inputs are blue, the constant node is gray, "
        "additions are green, and multiplications are orange."
    )


if __name__ == "__main__":
    main()
