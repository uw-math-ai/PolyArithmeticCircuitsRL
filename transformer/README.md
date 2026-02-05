# Transformer: Polynomial -> Circuit Translator

## What It Does
The `transformer/` package trains a character-level seq2seq Transformer that learns to translate
polynomial strings into equivalent circuit (SymPy-compatible) expression strings. The model
is trained on examples built from game-board JSONL dumps that encode algebraic operation
graphs.

At a high level:
- **Inputs**: polynomial strings like `x^2 + 2*x + 1`
- **Outputs**: a circuit-style expression string that is algebraically equivalent and suitable
  for SymPy parsing
- **Learning**: a Transformer is trained to map input strings to output strings
Note: Game boards now seed the constant `1` by default, so polynomials with constants are supported.

Key building blocks:
- **Board generation**: `build_training_data.py` can generate JSONL boards.
- **Dataset assembly**: `polynomial_to_circuit.py` and `generator.py` build `(poly, target)` pairs.
- **Model + tokenizer**: `polynomial_to_circuit.py` defines a char tokenizer and seq2seq Transformer.
- **Training**: `train.py` trains and saves a checkpoint.
- **Evaluation**: `eval.py` measures held-out accuracy and algebraic equivalence on unseen polynomials.
- **Inference**: `infer.py` loads a checkpoint and decodes a circuit string from a polynomial input.

## Example Scenario
Suppose you have a polynomial:
```
x^2 + 2*x + 1
```
You want a circuit-style expression that is algebraically equivalent so you can compare
different circuit constructions or feed it into downstream tooling. The Transformer is trained
on many examples of polynomial-to-circuit pairs and can infer a compatible expression, for
example:
```
(x+1)*(x+1)
```
The exact output string can differ across runs/checkpoints, but evaluation uses SymPy to check
algebraic equivalence rather than strict string equality.

## How To Run
You can run the whole pipeline, or run each step manually. All commands assume you are at
the repo root.

### Option A: End-to-End Pipeline
This generates a board, trains, and evaluates in one go.
```
python -m transformer.pipeline \
  --steps 4 \
  --num-vars 1 \
  --checkpoint transformer_checkpoints/board_C4.pt
```
Inputs:
- `--steps`: board complexity `C` (integer)
- `--num-vars`: number of variables (1 or 2)
- `--checkpoint`: output path for the trained model

Optional inputs:
- `--max-complexity`: max shortest-path length for examples
- `--include-non-multipath`: include nodes without multiple paths
- `--limit`: cap the number of training examples
- `--epochs`, `--batch-size`, `--lr`: training hyperparameters
- `--device`: force `cpu` or `cuda`
- `--metrics-out`: JSONL metrics output
- `--plot-out`: training plot PNG
- `--unseen-samples`, `--unseen-steps`, `--unseen-max-coeff`, `--unseen-seed`: unseen eval controls
- `--no-constant`: disable seeding the constant `1` node

### Option B: Manual Steps
#### 1) Generate a board (if you don’t already have one)
```
python -m transformer.build_training_data \
  --steps 4 \
  --num-vars 1 \
  --output-dir transformer/boards \
  --prefix game_board_C4_V1
```
Optional:
- `--no-constant`: disable seeding the constant `1` node
Outputs:
- `transformer/boards/game_board_C4_V1.nodes.jsonl`
- `transformer/boards/game_board_C4_V1.edges.jsonl`
- `transformer/boards/game_board_C4_V1.analysis.jsonl`

#### 2) Train
```
python -m transformer.train \
  --board-dir transformer/boards \
  --prefix game_board_C4_V1 \
  --epochs 10 \
  --batch-size 32 \
  --lr 3e-4 \
  --output transformer_checkpoints/board_C4.pt
```
Inputs:
- `--board-dir` and `--prefix`: locate the board JSONL files
- `--epochs`, `--batch-size`, `--lr`: training hyperparameters
- `--output`: checkpoint output path

Optional inputs:
- `--max-complexity`: max shortest-path length for dataset examples
- `--include-non-multipath`: include nodes without multiple paths
- `--limit`: cap the number of training examples
- `--val-split`: fraction of examples for validation
- `--metrics-out`: JSONL metrics output
- `--plot-out`: training plot PNG
- `--device`: force `cpu` or `cuda`
- `--no-constant`: disable seeding the constant `1` node when auto-generating a board

#### 3) Evaluate
```
python -m transformer.eval \
  --board-dir transformer/boards \
  --prefix game_board_C4_V1 \
  --checkpoint transformer_checkpoints/board_C4.pt \
  --num-vars 1 \
  --unseen-samples 200 \
  --unseen-steps 4
```
Inputs:
- `--checkpoint`: trained model path
- `--num-vars`: number of variables in unseen generation
- `--unseen-samples`: number of random polynomials to test
- `--unseen-steps`: complexity for unseen polynomials

Optional inputs:
- `--max-complexity`: max shortest-path length for dataset examples
- `--include-non-multipath`: include nodes without multiple paths
- `--limit`: cap the number of evaluation examples
- `--device`: force `cpu` or `cuda`
- `--unseen-max-coeff`, `--unseen-seed`: unseen generation controls
- `--no-constant`: disable seeding the constant `1` node when auto-generating a board

#### 4) Inference
```
python -m transformer.infer \
  --poly "x^2 + 2*x + 1" \
  --checkpoint transformer_checkpoints/board_C4.pt
```
Inputs:
- `--poly`: polynomial string (use `^` for exponent)
- `--checkpoint`: trained model path

Optional inputs:
- `--max-len`: max decode length
- `--auto-map-vars`: map `x,y,z,...` to `x0,x1,x2` if needed
- `--device`: force `cpu` or `cuda`
