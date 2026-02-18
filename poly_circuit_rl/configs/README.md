# configs/ - Hyperparameter reference

`default.yaml` is a reference template. Training uses values from `Config` in `poly_circuit_rl/config.py` unless explicitly overridden in code/CLI.

Use `default.yaml` for experiment notes and quick copy/paste, but treat `Config` as the source of truth for runtime defaults.

## Parameter groups

### Environment
- `n_vars`, `max_ops`, `L`, `max_nodes`, `m`, `eval_low`, `eval_high`
- `step_cost`: base penalty for `ADD`/`MUL`
- `shaping_coeff`: eval-distance shaping bonus strength
- `eval_norm_scale`: tanh normalization scale for eval vectors in observations
- `max_episode_steps`: optional hard cap on episode length (`None` uses derived default)

### Transformer
- `d_pos`, `d_model`, `n_heads`, `n_layers`, `dropout`

### DQN
- `lr`, `gamma`, `batch_size`, `buffer_size`
- `eps_start`, `eps_end`, `eps_decay_steps`
- `target_update_tau`, `train_freq`, `learning_starts`

### HER
- `her_k`: number of relabeled future-goal samples per transition

### Curriculum
- `curriculum_levels`: default `(1,2,3,4,5,6)`
- `curriculum_window`, `curriculum_threshold`

### Training
- `total_steps`, `eval_every`, `eval_episodes`, `seed`, `log_dir`

### Sampling
- `interesting_ratio`: interesting/random mix at higher curriculum levels
- `auto_interesting`: enable fallback auto-generation when no JSONL is provided
- `gen_max_graph_nodes`: auto-generation graph size cap
- `gen_max_successors`: auto-generation branching cap

## Notes

- `scripts/train.py` exposes only a subset of config fields as CLI flags.
- For full control (for example custom curriculum arrays), instantiate `Config(...)` directly in Python.
