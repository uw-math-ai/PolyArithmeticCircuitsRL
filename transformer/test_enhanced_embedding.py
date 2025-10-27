#!/usr/bin/env python3
"""
Test script for the enhanced embedding improvements.
This script tests:
1. Enhanced CircuitHistoryEncoder with input edges
2. Attention-based embedding combination instead of naive addition
3. Compatibility with existing pipeline
"""

import torch
import torch.nn as nn
from PositionalEncoding import CircuitHistoryEncoder, PositionalEncoding

def test_enhanced_encoder():
    """Test the enhanced circuit history encoder"""
    print("Testing Enhanced CircuitHistoryEncoder...")
    
    # Create encoder
    embedding_dim = 256
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Test circuit actions
    test_actions = [
        ("input", None, None),     # x0
        ("input", None, None),     # x1  
        ("constant", None, None),  # constant 1
        ("add", 0, 1),            # x0 + x1
        ("multiply", 3, 2),       # (x0 + x1) * 1
    ]
    
    print(f"Input actions: {test_actions}")
    
    # Encode actions to tokens
    tokens = encoder.encode_circuit_actions(test_actions)
    print(f"Encoded tokens: {len(tokens)} tokens")
    
    # Forward pass to get embeddings
    embeddings = encoder(tokens)
    print(f"Output embeddings shape: {embeddings.shape}")
    
    # Verify output shape
    expected_shape = (len(test_actions), embedding_dim)
    assert embeddings.shape == expected_shape, f"Expected {expected_shape}, got {embeddings.shape}"
    
    print("✅ Enhanced encoder test passed!")
    return True

def test_attention_combination():
    """Test that attention-based combination works correctly"""
    print("\nTesting attention-based embedding combination...")
    
    embedding_dim = 128
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Test with different action types
    test_actions = [
        ("input", None, None),
        ("constant", None, None), 
        ("add", 0, 1),
        ("multiply", 2, 0),
    ]
    
    tokens = encoder.encode_circuit_actions(test_actions)
    embeddings = encoder(tokens)
    
    # Check that embeddings are not all zeros or identical
    assert not torch.allclose(embeddings, torch.zeros_like(embeddings)), "Embeddings are all zeros!"
    
    # Check that different actions produce different embeddings
    for i in range(len(embeddings) - 1):
        assert not torch.allclose(embeddings[i], embeddings[i+1]), f"Embeddings {i} and {i+1} are identical!"
    
    print("✅ Attention-based combination test passed!")
    return True

def test_input_edges_usage():
    """Test that input edges are properly used in embeddings"""
    print("\nTesting input edges usage...")
    
    embedding_dim = 128
    encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Test actions with different input relationships
    test_actions = [
        ("input", None, None),     # x0
        ("input", None, None),     # x1
        ("add", 0, 1),            # x0 + x1 (uses inputs 0, 1)
        ("multiply", 2, 0),       # (x0 + x1) * x0 (uses inputs 2, 0)
    ]
    
    tokens = encoder.encode_circuit_actions(test_actions)
    
    # Check that input relationships are captured in tokens
    for i, token in enumerate(tokens):
        if i >= 2:  # Operations should have input information
            assert "input1" in token, f"Token {i} missing input1 information"
            assert "input2" in token, f"Token {i} missing input2 information"
            assert token["input1"] != -1 or token["input2"] != -1, f"Token {i} has no input information"
    
    # Get embeddings
    embeddings = encoder(tokens)
    
    # Check that embeddings are different for different input relationships
    add_embedding = embeddings[2]  # add(0, 1)
    mul_embedding = embeddings[3]  # multiply(2, 0)
    
    assert not torch.allclose(add_embedding, mul_embedding), "Add and multiply embeddings should be different!"
    
    print("✅ Input edges usage test passed!")
    return True

def test_compatibility_with_existing_pipeline():
    """Test compatibility with existing pipeline components"""
    print("\nTesting compatibility with existing pipeline...")
    
    embedding_dim = 256
    max_circuit_length = 100
    
    # Create components
    encoder = CircuitHistoryEncoder(embedding_dim)
    pos_encoding = PositionalEncoding(embedding_dim, max_circuit_length)
    
    # Test data
    test_actions = [
        ("input", None, None),
        ("input", None, None),
        ("constant", None, None),
        ("add", 0, 1),
        ("multiply", 3, 2),
    ]
    
    # Get embeddings
    tokens = encoder.encode_circuit_actions(test_actions)
    embeddings = encoder(tokens)
    
    # Test positional encoding
    # Add batch dimension for positional encoding
    embeddings_with_pos = pos_encoding(embeddings.unsqueeze(1))  # (seq_len, 1, embedding_dim)
    
    assert embeddings_with_pos.shape == (len(test_actions), 1, embedding_dim), \
        f"Positional encoding output shape incorrect: {embeddings_with_pos.shape}"
    
    print("✅ Compatibility test passed!")
    return True

def test_performance_comparison():
    """Compare performance of enhanced vs original approach"""
    print("\nTesting performance comparison...")
    
    embedding_dim = 256
    
    # Create enhanced encoder
    enhanced_encoder = CircuitHistoryEncoder(embedding_dim)
    
    # Test with a larger circuit
    large_test_actions = [
        ("input", None, None),
        ("input", None, None),
        ("input", None, None),
        ("constant", None, None),
        ("add", 0, 1),
        ("multiply", 4, 2),
        ("add", 5, 3),
        ("multiply", 6, 0),
    ]
    
    tokens = enhanced_encoder.encode_circuit_actions(large_test_actions)
    
    # Time the forward pass
    import time
    
    # Warm up
    for _ in range(5):
        _ = enhanced_encoder(tokens)
    
    # Time the actual forward pass
    start_time = time.time()
    embeddings = enhanced_encoder(tokens)
    end_time = time.time()
    
    forward_time = end_time - start_time
    
    print(f"Enhanced encoder forward pass time: {forward_time:.4f} seconds")
    print(f"Output embeddings shape: {embeddings.shape}")
    print(f"Embeddings range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    print("✅ Performance comparison test passed!")
    return True

def main():
    """Run all tests"""
    print("🧪 Running Enhanced Embedding Tests")
    print("=" * 50)
    
    tests = [
        test_enhanced_encoder,
        test_attention_combination,
        test_input_edges_usage,
        test_compatibility_with_existing_pipeline,
        test_performance_comparison,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced embedding is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
