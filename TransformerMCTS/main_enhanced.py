import torch
import os
import argparse
import traceback
from generator import generate_monomials_with_additive_indices
import numpy as np

# Import all functionality from the enhanced models file
from enhanced_models import AdvancedCircuitBuilder, parse_polynomial
from enhanced_models import train_supervised, train_reinforcement, evaluate_model, simplify_polynomial
from enhanced_models import Config, CircuitDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Polynomial Simplification with AlphaZero')
    parser.add_argument('--variables', type=int, default=3, help='Number of variables')
    parser.add_argument('--complexity', type=int, default=10, help='Maximum complexity')
    parser.add_argument('--train_size', type=int, default=5000, help='Training dataset size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--rl_episodes', type=int, default=2000, help='Number of RL episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for neural networks')
    parser.add_argument('--mode', choices=['supervised', 'reinforcement', 'both'], default='both',
                       help='Training mode: supervised, reinforcement, or both')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model on test data')
    parser.add_argument('--polynomial', type=str, default=None, help='Input polynomial to simplify (e.g., "x^2+2xy+y^2")')
    parser.add_argument('--simulations', type=int, default=500, help='Number of MCTS simulations for simplification')
    parser.add_argument('--temperature', type=float, default=0.01, help='Temperature for action selection in MCTS')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    # Print device information
    print(f"Using device: {device}")
    
    # Create configuration with argument overrides
    config = Config()
    config.n_variables = args.variables
    config.max_complexity = args.complexity
    config.train_size = args.train_size
    config.epochs = args.epochs
    config.rl_episodes = args.rl_episodes
    config.batch_size = args.batch_size 
    config.hidden_dim = args.hidden_dim
    config.num_simulations = args.simulations
    
    # Generate monomial indexing for polynomial representation
    index_to_monomial, monomial_to_index, _ = generate_monomials_with_additive_indices(
        config.n_variables, 
        config.polynomial_degree
    )
    
    # Calculate maximum vector size for polynomials
    max_vector_size = max(monomial_to_index.values()) + 1
    print(f"Maximum polynomial vector size: {max_vector_size}")
    
    # Initialize the advanced model
    print("Initializing model architecture...")
    model = AdvancedCircuitBuilder(config, max_vector_size).to(device)
    
    # Print model size info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Load pretrained model if specified
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        try:
            model.load_state_dict(torch.load(args.load_model, map_location=device))
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
    
    # If a polynomial is provided and model is loaded, just simplify without training
    if args.polynomial and args.load_model and os.path.exists(args.load_model):
        print(f"Using pre-trained model to simplify polynomial: {args.polynomial}")
        try:
            # Parse the polynomial into vector representation
            poly_vector = parse_polynomial(args.polynomial, index_to_monomial, monomial_to_index, config.n_variables)
            
            # Simplify the polynomial
            circuit, simplified_poly = simplify_polynomial(
                poly_vector, model, config, index_to_monomial, monomial_to_index
            )
            
            # Print result
            print("\nOriginal polynomial:", args.polynomial)
            print("Simplified circuit found!")
            
        except Exception as e:
            print(f"Error during polynomial simplification: {e}")
            traceback.print_exc()
        return  # Exit after simplification
    
    # Generate dataset for training/evaluation
    if args.mode in ['supervised', 'both'] or args.evaluate:
        print(f"Generating dataset with size {config.train_size}...")
        try:
            dataset = CircuitDataset(
                index_to_monomial, 
                monomial_to_index, 
                max_vector_size, 
                config, 
                size=config.train_size
            )
            print(f"Dataset generated with {len(dataset)} examples")
        except Exception as e:
            print(f"Error creating dataset: {e}")
            traceback.print_exc()
            dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, config, size=0)
    else:
        dataset = CircuitDataset(index_to_monomial, monomial_to_index, max_vector_size, config, size=0)
    
    # Train model with supervised learning
    if args.mode == 'supervised' or args.mode == 'both':
        print("Starting supervised training...")
        try:
            if len(dataset) > 0:
                model = train_supervised(model, dataset, config)
                
                # Save final supervised model
                model_path = f"final_supervised_model_n{config.n_variables}_C{config.max_complexity}.pt"
                torch.save(model.state_dict(), model_path)
                print(f"Supervised model saved to {model_path}")
            else:
                print("No training data available for supervised learning.")
        except Exception as e:
            print(f"Error during supervised training: {e}")
            traceback.print_exc()
    
    # Train model with reinforcement learning
    if args.mode == 'reinforcement' or args.mode == 'both':
        print("Starting reinforcement learning...")
        try:
            model = train_reinforcement(model, index_to_monomial, monomial_to_index, config, max_vector_size)
            
            # Save final RL model
            model_path = f"final_rl_model_n{config.n_variables}_C{config.max_complexity}.pt"
            torch.save(model.state_dict(), model_path)
            print(f"Reinforcement learning model saved to {model_path}")
        except Exception as e:
            print(f"Error during reinforcement learning: {e}")
            traceback.print_exc()
    
    # Evaluate model if requested
    if args.evaluate:
        print("Evaluating model...")
        try:
            if len(dataset) > 0:
                accuracy, mse, similarity = evaluate_model(model, dataset, config, index_to_monomial, monomial_to_index)
                print(f"Evaluation results: Accuracy={accuracy:.2f}%, MSE={mse:.4f}, Similarity={similarity:.4f}")
            else:
                print("No data available for evaluation.")
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
    
    # Simplify a specific polynomial if provided (when not using a pre-trained model)
    if args.polynomial and not (args.load_model and os.path.exists(args.load_model)):
        print(f"Simplifying polynomial: {args.polynomial}")
        try:
            poly_vector = parse_polynomial(args.polynomial, index_to_monomial, monomial_to_index, config.n_variables)
            circuit, simplified_poly = simplify_polynomial(poly_vector, model, config, index_to_monomial, monomial_to_index)
            
            print("\nOriginal polynomial:", args.polynomial)
            print("Simplified arithmetic circuit found!")
        except Exception as e:
            print(f"Error during polynomial simplification: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()