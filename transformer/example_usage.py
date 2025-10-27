#!/usr/bin/env python3
"""
Example usage of the enhanced embedding system.
This shows how to use the improved CircuitHistoryEncoder in your existing pipeline.
"""

import torch
import torch.nn as nn
from PositionalEncoding import CircuitHistoryEncoder, PositionalEncoding

def example_basic_usage():
    """Basic usage example"""
    print("🔧 Basic Usage Example")
    print("-" * 30)
    
    # Create enhanced encoder
    embedding_dim = 256
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Example circuit actions
    circuit_actions = [
        ("input", None, None),     # x0
        ("input", None, None),     # x1
        ("constant", None, None),  # constant 1
        ("add", 0, 1),            # x0 + x1
        ("multiply", 3, 2),       # (x0 + x1) * 1
    ]
    
    print(f"Circuit actions: {circuit_actions}")
    
    # Encode to tokens
    tokens = encoder.encode_circuit_actions(circuit_actions)
    print(f"Generated {len(tokens)} tokens")
    
    # Get embeddings
    embeddings = encoder(tokens)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Show that input relationships are captured
    print("\nToken details:")
    for i, token in enumerate(tokens):
        print(f"  Token {i}: type={token['type']}, inputs=({token['input1']}, {token['input2']})")
    
    return embeddings

def example_with_positional_encoding():
    """Example with positional encoding"""
    print("\n🔧 Example with Positional Encoding")
    print("-" * 40)
    
    embedding_dim = 256
    max_circuit_length = 100
    
    # Create components
    encoder = CircuitHistoryEncoder(embedding_dim)
    pos_encoding = PositionalEncoding(embedding_dim, max_circuit_length)
    
    # Example circuit
    circuit_actions = [
        ("input", None, None),
        ("input", None, None),
        ("constant", None, None),
        ("add", 0, 1),
        ("multiply", 3, 2),
        ("add", 4, 0),
    ]
    
    # Get embeddings
    tokens = encoder.encode_circuit_actions(circuit_actions)
    embeddings = encoder(tokens)
    
    # Apply positional encoding
    embeddings_with_pos = pos_encoding(embeddings.unsqueeze(1))  # Add batch dim
    
    print(f"Original embeddings shape: {embeddings.shape}")
    print(f"With positional encoding: {embeddings_with_pos.shape}")
    
    return embeddings_with_pos

def example_batch_processing():
    """Example of batch processing"""
    print("\n🔧 Batch Processing Example")
    print("-" * 35)
    
    embedding_dim = 256
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Multiple circuits in a batch
    batch_circuits = [
        [
            ("input", None, None),
            ("constant", None, None),
            ("add", 0, 1),
        ],
        [
            ("input", None, None),
            ("input", None, None),
            ("multiply", 0, 1),
        ]
    ]
    
    # Process each circuit
    batch_embeddings = []
    for i, circuit in enumerate(batch_circuits):
        tokens = encoder.encode_circuit_actions(circuit)
        embeddings = encoder(tokens)
        batch_embeddings.append(embeddings)
        print(f"Circuit {i}: {len(circuit)} actions -> {embeddings.shape}")
    
    # Pad to same length for batch processing
    max_len = max(emb.shape[0] for emb in batch_embeddings)
    padded_embeddings = []
    
    for emb in batch_embeddings:
        if emb.shape[0] < max_len:
            padding = torch.zeros(max_len - emb.shape[0], embedding_dim)
            padded_emb = torch.cat([emb, padding], dim=0)
        else:
            padded_emb = emb
        padded_embeddings.append(padded_emb)
    
    # Stack into batch
    batch_tensor = torch.stack(padded_embeddings, dim=1)  # (seq_len, batch_size, embedding_dim)
    print(f"Batch tensor shape: {batch_tensor.shape}")
    
    return batch_tensor

def example_comparison_with_original():
    """Show the difference between original and enhanced approach"""
    print("\n🔧 Comparison: Original vs Enhanced")
    print("-" * 45)
    
    embedding_dim = 256
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Test circuit with clear input relationships
    circuit_actions = [
        ("input", None, None),     # x0
        ("input", None, None),     # x1
        ("add", 0, 1),            # x0 + x1 (depends on inputs 0, 1)
        ("multiply", 2, 0),       # (x0 + x1) * x0 (depends on inputs 2, 0)
    ]
    
    tokens = encoder.encode_circuit_actions(circuit_actions)
    embeddings = encoder(tokens)
    
    print("Enhanced embeddings capture input relationships:")
    for i, (action, token, emb) in enumerate(zip(circuit_actions, tokens, embeddings)):
        if action[0] in ['add', 'multiply']:
            print(f"  {action[0]}({action[1]}, {action[2]}): inputs=({token['input1']}, {token['input2']})")
            print(f"    Embedding range: [{emb.min():.3f}, {emb.max():.3f}]")
    
    # Show that different input relationships produce different embeddings
    add_embedding = embeddings[2]  # add(0, 1)
    mul_embedding = embeddings[3]  # multiply(2, 0)
    
    similarity = torch.cosine_similarity(add_embedding, mul_embedding, dim=0)
    print(f"\nCosine similarity between add and multiply embeddings: {similarity:.3f}")
    
    return embeddings

def main():
    """Run all examples"""
    print("🚀 Enhanced Embedding Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_usage()
        example_with_positional_encoding()
        example_batch_processing()
        example_comparison_with_original()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nThe enhanced embedding is ready to use in your pipeline.")
        print("Simply import CircuitHistoryEncoder from PositionalEncoding.py")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
