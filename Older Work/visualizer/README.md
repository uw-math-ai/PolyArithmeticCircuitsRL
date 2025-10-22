**Overview**
- Web visualizer for the arithmetic circuit RL model. Generates a random polynomial, evaluates the trained model to build a circuit, displays factorization, and renders the circuit tree (+, *).

**Run**
- Install deps (CPU example): `pip install -r requirements.txt`
- Start app: `streamlit run visualizer/app.py`

**Notes**
- The app loads `transformer/ppo_model_n{N}_C{C}_curriculum.pt` by default. Adjust Config in `transformer/fourthGen.py` if needed.
- If no PPO checkpoint exists, it will try the best supervised checkpoint `best_supervised_model_n{N}_C{C}.pt`.
- For GPU, install CUDA-enabled PyTorch and matching PyG wheels (see `transformer/README.md`).

