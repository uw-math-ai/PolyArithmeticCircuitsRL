**Overview**
- This folder contains the Transformer-based reinforcement learning system used to learn arithmetic circuits that exactly compute target polynomials. The agent incrementally builds a circuit by choosing actions (add/multiply two existing nodes). It combines a GNN to embed the current circuit state, a Transformer decoder to condition on the target polynomial and circuit history, and policy/value heads trained with PPO.

**Key Ideas**
- State encoder: a GNN embeds the current arithmetic circuit graph; a small sequence encoder embeds the action history; a linear layer embeds the target polynomial vector.
- Policy/value: a Transformer decoder attends over the target polynomial and history, then predicts the next action via a masked policy head; a value head estimates the state value.
- Training pipeline: supervised pretraining on synthetic circuits, followed by Proximal Policy Optimization (PPO) with curriculum learning and reward shaping to improve stability and search quality.

**Files**
- `transformer/fourthGen.py`: Main training pipeline and model. Includes:
  - `Config`: hyperparameters (model sizes, PPO, curriculum) for N=3, C=5 by default.
  - `CircuitBuilder`: GNN + Transformer + policy/value heads; `get_action_and_value` for PPO.
  - `CircuitDataset`: supervised samples from synthetic circuits.
  - `train_supervised`, `train_ppo`, `evaluate_model`, and `main`.
- `transformer/SupervisedTransformer.py`: Earlier baseline with larger model and REINFORCE-style RL. Kept for reference; use `fourthGen.py` for training.
- `transformer/State.py`: RL environment (`Game`) for circuit construction, action masking, reward shaping, SymPy checks, and Torch Geometric graph building.
- `transformer/generator.py`: Synthetic data utilities. Monomial additive indexing, random circuit generation, vector ops for add/multiply, and dataset trimming.
- `transformer/PositionalEncoding.py`: `CircuitHistoryEncoder` and sinusoidal `PositionalEncoding` for sequence inputs to the Transformer.
- `transformer/utils.py`: Helpers such as `encode_action` (commutative pair + op to index) and `vector_to_sympy`.
- `transformer/test_model.py`: Inference and interactive testing. Loads a trained model and uses a hybrid top-w tree search (and beam search utility) to find a circuit for a user-provided polynomial.
- `transformer/ppo_model_n3_C5_curriculum.pt`: Pretrained checkpoint for N=3 variables, complexity C=5 (curriculum-trained PPO).
- `transformer/ReadmePlease.txt`: Internal notes with a changelog-like summary of the PPO transition and reward shaping.

**Environment**
- Python 3.9+ recommended.
- Packages: `torch`, `torch_geometric`, `sympy`, `numpy`, `tqdm`.
- Install from `requirements.txt` for a baseline: `pip install -r requirements.txt`.
- Torch Geometric requires wheels that match your PyTorch and CUDA/CPU build. See CPU/GPU notes below.

**CPU vs GPU**
- GPU is recommended for PPO training; CPU is sufficient for small tests and inference with reduced settings.
- The environment keeps game logic on CPU by default; the model runs on CUDA if available. This mixed setup is already handled in `fourthGen.py`.

Install examples (choose one):
- CPU-only
  - Install PyTorch CPU build:
    - `pip install --index-url https://download.pytorch.org/whl/cpu torch`
  - Install matching PyG wheels (adjust torch version in the URL if needed):
    - `pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html`
  - Then: `pip install torch_geometric sympy numpy tqdm`

- NVIDIA GPU (example: CUDA 12.1)
  - Install PyTorch with CUDA:
    - `pip install --index-url https://download.pytorch.org/whl/cu121 torch`
  - Install matching PyG wheels (note the torch and CUDA tags in the URL):
    - `pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html`
  - Then: `pip install torch_geometric sympy numpy tqdm`

- Apple Silicon (MPS)
  - Install the standard PyTorch build with MPS support (macOS 12.3+):
    - `pip install torch`
  - PyG GPU acceleration on MPS is limited; prefer CPU wheels as above or run CPU-only.

Verify your setup
- Check Torch and CUDA availability:
  - `python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cuda_ver', getattr(torch.version, 'cuda', None))"`
- Check PyG version:
  - `python -c "import torch_geometric as tg; print('pyg', tg.__version__)"`

Device tips
- Force CPU: set `CUDA_VISIBLE_DEVICES=""` or run with `torch.device('cpu')`.
- If you change `Config` to larger models or higher PPO batch sizes on CPU, expect longer runtimes; you can reduce `steps_per_batch`, `ppo_iterations`, and model depth.

**Quick Start**
- Train (supervised → PPO):
  - From repo root: `python transformer/fourthGen.py`
  - Artifacts:
    - Best supervised: `best_supervised_model_n{N}_C{C}.pt`
    - PPO model: `ppo_model_n{N}_C{C}_curriculum.pt`
- Use the pretrained PPO model (included):
  - From repo root: `python transformer/test_model.py`
  - When prompted, enter a polynomial using variables `x0..x{N-1}`. Examples for N=3:
    - `x0 + x1`
    - `x0*(x1 + 1)`
    - `x0**2 + 2*x0*x1 + x1**2`
  - The script runs a hybrid top-w tree search guided by the model to construct a circuit. It prints each step and the resulting expression.

**Configuration**
- Edit `transformer/fourthGen.py:Config` to change:
  - Problem size: `n_variables` (N), `max_complexity` (C), `mod` (coefficient modulus), dataset sizes.
  - Model: `embedding_dim`, `hidden_dim`, GNN/Transformer depth, heads, dropout.
  - PPO: learning rate, epochs, minibatch size, `ppo_clip`, `gamma`, `lambda_gae`, `ent_coef`, `action_temperature`.
  - Curriculum: `complexity_threshold`, `complexity_window` for increasing target complexity across PPO iterations.

**How It Works**
- Monomial indexing: `generator.generate_monomials_with_additive_indices` builds an index where exponent addition maps to index addition. A polynomial is a dense coefficient vector over these indices (mod `mod`).
- State and actions:
  - Base nodes: N variable nodes and one constant-1 node.
  - Action space: choose op in {add, multiply} and an unordered pair of existing nodes. `utils.encode_action` maps to a stable action index; `State._action_mask` enables only valid actions.
- Model forward:
  - GNN embeds the current circuit graph; `CircuitHistoryEncoder` embeds history; a linear layer embeds the target polynomial vector.
  - A Transformer decoder attends over target + history and emits a single output token, from which policy logits and a scalar value are predicted.
- Rewards and curriculum:
  - Reward shaping encourages reducing L1 distance to the target vector, penalizes steps, and gives a success bonus; penalties on failure at max steps.
  - PPO collects trajectories at a given complexity and gradually increases complexity when recent success rate passes a threshold.

**Troubleshooting**
- Torch Geometric install issues: ensure your PyTorch and CUDA versions match the recommended wheels. See the official docs if `pip install torch_geometric` fails.
- Model not found: `test_model.py` looks for `ppo_model_n{N}_C{C}_curriculum.pt` in `transformer/`. Use the provided `ppo_model_n3_C5_curriculum.pt` or train with `fourthGen.py`.
- Input syntax: `test_model.py` uses SymPy parsing with implicit multiplication; `**` or `^` for exponents are accepted (e.g., `x0**2`, `x0^2`). Variables are `x0, x1, ...` up to `x{N-1}`.
- Performance: PPO is compute-heavy. A GPU is recommended; lower `steps_per_batch`, `ppo_iterations`, or model depth for quicker experiments.

**Citations / Credit**
- PPO: Schulman et al., “Proximal Policy Optimization Algorithms” (2017).
- Graph/sequence backbone inspired by standard GNN + Transformer decoding for structured prediction.
