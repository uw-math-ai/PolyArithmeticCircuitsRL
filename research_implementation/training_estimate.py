"""
Training Time and Memory Estimation for PolyArithmeticCircuitsRL

This script estimates the computational requirements for a full training run
on a local CPU machine based on the current configuration.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import time

# Import configuration
try:
    from fourthGen import Config, CircuitBuilder
    from generator import generate_monomials_with_additive_indices
    IMPORTS_OK = True
except:
    IMPORTS_OK = False
    print("Warning: Could not import modules, using estimated values")


class TrainingEstimator:
    """Estimate training time and memory requirements."""
    
    def __init__(self):
        self.config = Config() if IMPORTS_OK else self._mock_config()
        
        # Calculate model size parameters
        n, d = self.config.n_variables, self.config.max_complexity * 2
        if IMPORTS_OK:
            index_to_monomial, _, _ = generate_monomials_with_additive_indices(n, d)
            self.max_vector_size = len(index_to_monomial)
        else:
            # Estimate polynomial vector size
            # For n=3, d=10: approximately (d+n choose n) monomials
            from math import comb
            self.max_vector_size = comb(d + n, n)
        
        # Calculate action space size
        max_nodes = self.config.n_variables + self.config.max_complexity + 1
        self.max_actions = (max_nodes * (max_nodes + 1) // 2) * 2
        
        print(f"Configuration Analysis:")
        print(f"  Variables: {self.config.n_variables}")
        print(f"  Max complexity: {self.config.max_complexity}")
        print(f"  Max polynomial vector size: {self.max_vector_size}")
        print(f"  Max action space size: {self.max_actions}")
        print(f"  Embedding dimension: {self.config.embedding_dim}")
        print(f"  Hidden dimension: {self.config.hidden_dim}")
        print()
    
    def _mock_config(self):
        """Mock config if imports fail."""
        class MockConfig:
            def __init__(self):
                self.n_variables = 3
                self.max_complexity = 5
                self.hidden_dim = 256
                self.embedding_dim = 256
                self.num_gnn_layers = 3
                self.num_transformer_layers = 6
                self.transformer_heads = 4
                self.train_size = 10000
                self.test_size = 2000
                self.epochs = 50
                self.batch_size = 128
                self.ppo_iterations = 2000
                self.steps_per_batch = 4096
                self.ppo_epochs = 10
                self.ppo_minibatch_size = 128
        return MockConfig()
    
    def estimate_model_parameters(self) -> Tuple[int, float]:
        """
        Estimate the number of parameters in the model.
        
        Returns:
            (num_parameters, memory_mb)
        """
        config = self.config
        
        # GNN parameters
        gnn_params = 0
        # First layer: 4 -> hidden_dim
        gnn_params += 4 * config.hidden_dim + config.hidden_dim  # weights + bias
        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(config.num_gnn_layers - 2):
            gnn_params += config.hidden_dim * config.hidden_dim + config.hidden_dim
        # Last layer: hidden_dim -> embedding_dim
        gnn_params += config.hidden_dim * config.embedding_dim + config.embedding_dim
        # Layer norms
        gnn_params += (config.num_gnn_layers - 1) * config.hidden_dim * 2  # weight + bias
        gnn_params += config.embedding_dim * 2  # final layer norm
        
        # Polynomial embedding
        poly_embed_params = self.max_vector_size * config.embedding_dim + config.embedding_dim
        
        # Transformer parameters
        transformer_params = 0
        # Each decoder layer has: self-attention + cross-attention + feedforward
        d_model = config.embedding_dim
        n_heads = config.transformer_heads
        d_ff = config.hidden_dim
        
        for _ in range(config.num_transformer_layers):
            # Self-attention: Q, K, V projections + output projection
            transformer_params += 4 * (d_model * d_model + d_model)
            # Cross-attention: Q, K, V projections + output projection  
            transformer_params += 4 * (d_model * d_model + d_model)
            # Feedforward: two linear layers
            transformer_params += d_model * d_ff + d_ff  # first layer
            transformer_params += d_ff * d_model + d_model  # second layer
            # Layer norms (3 per layer)
            transformer_params += 3 * (d_model * 2)
        
        # Action and value heads
        action_head_params = config.embedding_dim * self.max_actions + self.max_actions
        value_head_params = config.embedding_dim * 1 + 1
        
        # Output token
        output_token_params = config.embedding_dim
        
        # Circuit history encoder (estimate)
        circuit_encoder_params = config.embedding_dim * 100  # rough estimate
        
        total_params = (gnn_params + poly_embed_params + transformer_params + 
                       action_head_params + value_head_params + output_token_params +
                       circuit_encoder_params)
        
        # Memory in MB (4 bytes per float32 parameter)
        memory_mb = total_params * 4 / (1024 * 1024)
        
        return total_params, memory_mb
    
    def estimate_training_memory(self) -> Dict[str, float]:
        """
        Estimate memory usage during training.
        
        Returns:
            Dictionary with memory estimates in MB
        """
        model_params, model_memory = self.estimate_model_parameters()
        
        # Optimizer state (Adam stores momentum and velocity)
        optimizer_memory = model_memory * 2  # 2x model size for Adam
        
        # Gradients
        gradient_memory = model_memory
        
        # Supervised training batch memory
        batch_size = self.config.batch_size
        
        # Estimate activations memory per sample
        embedding_dim = self.config.embedding_dim
        max_seq_len = self.config.max_complexity  # rough estimate
        
        # GNN activations
        max_nodes = 20  # estimate
        gnn_activation_per_sample = max_nodes * embedding_dim * 4 / (1024 * 1024)
        
        # Transformer activations (very rough estimate)
        transformer_activation_per_sample = (max_seq_len * embedding_dim * 
                                           self.config.num_transformer_layers * 4) / (1024 * 1024)
        
        supervised_batch_memory = batch_size * (gnn_activation_per_sample + 
                                               transformer_activation_per_sample +
                                               self.max_vector_size * 4 / (1024 * 1024))
        
        # PPO memory (larger due to trajectory collection)
        ppo_batch_size = self.config.steps_per_batch
        ppo_batch_memory = ppo_batch_size * 0.1  # rough estimate per step in MB
        
        return {
            'model_parameters': model_memory,
            'optimizer_state': optimizer_memory,
            'gradients': gradient_memory,
            'supervised_batch': supervised_batch_memory,
            'ppo_batch': ppo_batch_memory,
            'total_supervised': model_memory + optimizer_memory + gradient_memory + supervised_batch_memory,
            'total_ppo': model_memory + optimizer_memory + gradient_memory + ppo_batch_memory
        }
    
    def estimate_training_time(self) -> Dict[str, float]:
        """
        Estimate training time on CPU.
        
        Returns:
            Dictionary with time estimates
        """
        config = self.config
        
        # Rough estimates based on model complexity
        # These are very approximate and depend heavily on CPU performance
        
        # Forward pass time per sample (seconds)
        # Transformer and GNN are computationally expensive
        forward_time_per_sample = 0.05  # 50ms per sample on modest CPU
        
        # Backward pass is roughly 2x forward pass
        backward_time_per_sample = forward_time_per_sample * 2
        
        # Supervised training
        samples_per_epoch = config.train_size
        batches_per_epoch = np.ceil(samples_per_epoch / config.batch_size)
        
        supervised_time_per_epoch = (batches_per_epoch * config.batch_size * 
                                   (forward_time_per_sample + backward_time_per_sample))
        supervised_total_time = supervised_time_per_epoch * config.epochs
        
        # PPO training (more complex due to environment interaction)
        # Each PPO iteration involves:
        # 1. Trajectory collection (forward passes only)
        # 2. Multiple update epochs (forward + backward)
        
        trajectory_time_per_iteration = config.steps_per_batch * forward_time_per_sample
        
        # PPO update phase
        ppo_samples_per_iteration = config.steps_per_batch
        ppo_batches_per_epoch = np.ceil(ppo_samples_per_iteration / config.ppo_minibatch_size)
        ppo_update_time_per_iteration = (config.ppo_epochs * ppo_batches_per_epoch * 
                                        config.ppo_minibatch_size * 
                                        (forward_time_per_sample + backward_time_per_sample))
        
        ppo_time_per_iteration = trajectory_time_per_iteration + ppo_update_time_per_iteration
        ppo_total_time = ppo_time_per_iteration * config.ppo_iterations
        
        return {
            'supervised_per_epoch_hours': supervised_time_per_epoch / 3600,
            'supervised_total_hours': supervised_total_time / 3600,
            'ppo_per_iteration_minutes': ppo_time_per_iteration / 60,
            'ppo_total_hours': ppo_total_time / 3600,
            'total_training_hours': (supervised_total_time + ppo_total_time) / 3600,
            'total_training_days': (supervised_total_time + ppo_total_time) / (3600 * 24)
        }
    
    def print_detailed_estimate(self):
        """Print comprehensive training estimates."""
        print("=" * 60)
        print("TRAINING TIME AND MEMORY ESTIMATION")
        print("=" * 60)
        
        # Model size
        num_params, model_mb = self.estimate_model_parameters()
        print(f"\nMODEL SIZE:")
        print(f"  Parameters: {num_params:,}")
        print(f"  Model memory: {model_mb:.1f} MB")
        
        # Memory requirements
        memory_est = self.estimate_training_memory()
        print(f"\nMEMORY REQUIREMENTS:")
        print(f"  Model parameters: {memory_est['model_parameters']:.1f} MB")
        print(f"  Optimizer state (Adam): {memory_est['optimizer_state']:.1f} MB")
        print(f"  Gradients: {memory_est['gradients']:.1f} MB")
        print(f"  Supervised batch: {memory_est['supervised_batch']:.1f} MB")
        print(f"  PPO batch: {memory_est['ppo_batch']:.1f} MB")
        print(f"  Total (supervised): {memory_est['total_supervised']:.1f} MB")
        print(f"  Total (PPO): {memory_est['total_ppo']:.1f} MB")
        
        # Time estimates
        time_est = self.estimate_training_time()
        print(f"\nTIME ESTIMATES (CPU):")
        print(f"  Supervised training:")
        print(f"    Per epoch: {time_est['supervised_per_epoch_hours']:.2f} hours")
        print(f"    Total ({self.config.epochs} epochs): {time_est['supervised_total_hours']:.1f} hours")
        print(f"  PPO training:")
        print(f"    Per iteration: {time_est['ppo_per_iteration_minutes']:.1f} minutes")
        print(f"    Total ({self.config.ppo_iterations} iterations): {time_est['ppo_total_hours']:.1f} hours")
        print(f"  TOTAL TRAINING TIME: {time_est['total_training_hours']:.1f} hours")
        print(f"  TOTAL TRAINING TIME: {time_est['total_training_days']:.1f} days")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if memory_est['total_ppo'] > 8000:  # 8GB
            print("  ⚠️  High memory usage - consider reducing batch sizes")
        if time_est['total_training_days'] > 7:
            print("  ⚠️  Very long training time - consider:")
            print("     - Reducing model size (embedding_dim, num_layers)")
            print("     - Reducing training iterations/epochs")
            print("     - Using GPU acceleration")
        if time_est['total_training_days'] > 30:
            print("  🚨  Extremely long training time - GPU strongly recommended")
        
        # System requirements
        print(f"\nSYSTEM REQUIREMENTS:")
        print(f"  Minimum RAM: {max(memory_est['total_ppo'], 4000):.0f} MB ({max(memory_est['total_ppo']/1024, 4):.1f} GB)")
        print(f"  Recommended RAM: {max(memory_est['total_ppo'] * 1.5, 8000):.0f} MB ({max(memory_est['total_ppo']*1.5/1024, 8):.1f} GB)")
        print(f"  Storage needed: ~500 MB (model checkpoints)")
        print(f"  CPU: Multi-core recommended (8+ cores for reasonable speed)")


def main():
    """Run training estimation."""
    estimator = TrainingEstimator()
    estimator.print_detailed_estimate()
    
    print(f"\n" + "=" * 60)
    print("NOTE: These are rough estimates. Actual performance depends on:")
    print("- CPU performance (clock speed, cores, cache)")
    print("- RAM speed and availability")
    print("- Python/PyTorch optimizations")
    print("- System load and other running processes")
    print("- Model convergence (may finish earlier if successful)")
    print("=" * 60)


if __name__ == "__main__":
    main()