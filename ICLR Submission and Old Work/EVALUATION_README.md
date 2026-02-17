# PPO Model Evaluation Script

## Overview
This script evaluates the trained PPO model (`ppo_model_n3_C6_curriculum.pt`) by testing it on 10 randomly generated polynomial targets with complexity 6 and 3 variables.

## File
- **Location**: `evaluate_ppo_model.py`
- **Model Path**: `ppo_model_n3_C6_curriculum.pt`
- **Results Output**: `evaluation_results_C6.json`

## Features

### What the Script Does
<<<<<<< HEAD
1. Generates 10 unique random polynomial targets with complexity 6 and 3 variables
=======
1. Generates 10 unique random polynomial targets with complexity 6 and 3 variables    // from where?
>>>>>>> 11b48741e682c6fc7ea309bcbc3750e60bf7594b
2. Creates reference circuits for each polynomial using the circuit generation algorithm
3. Loads the trained PPO model from the specified checkpoint
4. Evaluates the model on each polynomial:
   - Tests if the model can construct a circuit matching the target polynomial
   - Tracks the number of steps taken and the final polynomial constructed
   - Records success/failure for each test case
5. Saves detailed results to `evaluation_results_C6.json`

### Model Configuration
- **Number of Variables**: 3 (x0, x1, x2)
- **Max Complexity**: 6 (max 6 arithmetic operations)
- **Modulo**: 50 (coefficients mod 50)
- **Device**: Automatically uses CUDA if available, falls back to CPU

### Output Format

The results are saved in JSON format with the following structure:
```json
{
  "config": {
    "n_variables": 3,
    "max_complexity": 6,
    "mod": 50
  },
  "summary": {
    "total_tests": 10,
    "successes": 2,
    "success_rate": 0.2,
    "average_steps": 5.0
  },
  "results": [
    {
      "test_id": 1,
      "target_polynomial": "x0 + x1",
      "success": true,
      "steps": 1,
      "final_polynomial": "x0 + x1",
      "reference_circuit_length": 3
    },
    ...
  ]
}
```

## Usage

### Basic Usage
```bash
cd /home/ec2-user/DESKTOP/Naomi/PolyArithmeticCircuitsRL
python3 evaluate_ppo_model.py
```

### Requirements
- Python 3.8+
- PyTorch (GPU support recommended)
- SymPy
- NumPy
- PyTorch Geometric

All requirements are listed in `requirements.txt`

## Interpretation of Results

### Success Metrics
- **Success**: Model correctly found the target polynomial
- **Average Steps**: Mean number of steps (operations) the model took across all tests
- **Success Rate**: Percentage of tests where the model found the target

### Example Output
```
Test 1/10:
  Target polynomial: x0 + x1
  Reference circuit length: 3
  Success: ✓ YES
  Steps taken: 1
  Final polynomial: x0 + x1
```

## Implementation Details

### Key Components

1. **Target Polynomial Generation**: Uses `generate_random_circuit()` to create valid polynomial circuits with exactly the specified complexity level.

2. **Target Encoding**: The target polynomial is encoded using the reference circuit via `encode_actions_with_compact_encoder()` to provide the model with the polynomial structure information.

3. **Model Evaluation**: The `evaluate_on_polynomial()` function:
   - Creates a Game environment with the target polynomial
   - Steps through the model's action sampling until completion or max steps reached
   - Checks if the final polynomial matches the target
   - Returns success status and metrics

4. **Graph Representation**: The state is represented as a PyTorch Geometric graph where:
   - Nodes represent circuit nodes (inputs, constants, and operations)
   - Edges represent data dependencies
   - Node features include type (input/constant/operation) and normalized value

## Notes

- The model may not achieve 100% success rate due to the complexity of the task and the randomness of polynomial generation
- The evaluation uses the same device (CUDA/CPU) as the model was trained on
- Results are saved with a timestamp in the filename for tracking multiple evaluation runs
- The script uses a maximum of `max_complexity + 5` steps to allow some exploration beyond the minimum required complexity

## Troubleshooting

### Model File Not Found
Ensure the model path is correct:
```
/home/ec2-user/DESKTOP/Naomi/PolyArithmeticCircuitsRL/ppo_model_n3_C6_curriculum.pt
```

### CUDA Out of Memory
If you encounter CUDA memory errors, the script will automatically fall back to CPU mode.

### Slow Evaluation
- The evaluation uses CUDA by default (if available) for faster inference
- Each polynomial evaluation typically takes 1-2 seconds on GPU

## Future Improvements

Possible enhancements to the evaluation script:
- Test multiple complexity levels
- Variable number of variables (2-5)
- Compare with baseline/reference solutions
- Visualization of generated circuits
- Statistical analysis of success patterns
