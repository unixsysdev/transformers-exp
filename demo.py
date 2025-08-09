#!/usr/bin/env python3
"""
Demo script for Space-Time Optimized Transformer
Shows memory usage comparison and basic functionality
"""

import torch
import torch.nn as nn
import time
import tracemalloc
from transformer import SpaceTimeOptimizedTransformer
import matplotlib.pyplot as plt
import numpy as np

def measure_memory_and_time(model, input_ids, name="Model"):
    """Measure memory usage and inference time"""
    # Start memory tracking
    tracemalloc.start()
    
    # Warm up
    with torch.no_grad():
        _ = model(input_ids)
    
    # Actual measurement
    start_time = time.time()
    tracemalloc.reset_peak()
    
    with torch.no_grad():
        output = model(input_ids)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        'name': name,
        'time': end_time - start_time,
        'peak_memory_mb': peak / 1024 / 1024,
        'output_shape': output.shape
    }

def run_demo():
    print("🚀 Space-Time Optimized Transformer Demo")
    print("=" * 50)
    
    # Test different sequence lengths
    vocab_size = 5000
    batch_size = 2
    sequence_lengths = [64, 128, 256, 512]
    
    results = []
    
    for seq_len in sequence_lengths:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Create input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Standard model (without quantization for comparison)
        standard_model = SpaceTimeOptimizedTransformer(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=4,
            use_quantization=False
        )
        
        # Optimized model (with quantization and space-time optimizations)
        optimized_model = SpaceTimeOptimizedTransformer(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=8,
            n_layers=4,
            use_quantization=True
        )
        
        # Measure both models
        standard_result = measure_memory_and_time(standard_model, input_ids, f"Standard (seq={seq_len})")
        optimized_result = measure_memory_and_time(optimized_model, input_ids, f"Optimized (seq={seq_len})")
        
        results.extend([standard_result, optimized_result])
        
        print(f"  Standard  - Time: {standard_result['time']:.3f}s, Memory: {standard_result['peak_memory_mb']:.1f}MB")
        print(f"  Optimized - Time: {optimized_result['time']:.3f}s, Memory: {optimized_result['peak_memory_mb']:.1f}MB")
        
        memory_savings = (1 - optimized_result['peak_memory_mb'] / standard_result['peak_memory_mb']) * 100
        print(f"  💾 Memory savings: {memory_savings:.1f}%")
    
    # Plot results
    print(f"\n📊 Creating performance visualization...")
    create_performance_plot(results, sequence_lengths)
    
    print(f"\n✨ Demo completed! Check the generated plot for visual comparison.")

def create_performance_plot(results, sequence_lengths):
    """Create a plot comparing memory usage across sequence lengths"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract data for plotting
    standard_memory = []
    optimized_memory = []
    standard_time = []
    optimized_time = []
    
    for i in range(0, len(results), 2):
        standard = results[i]
        optimized = results[i + 1]
        
        standard_memory.append(standard['peak_memory_mb'])
        optimized_memory.append(optimized['peak_memory_mb'])
        standard_time.append(standard['time'])
        optimized_time.append(optimized['time'])
    
    # Memory usage plot
    x = np.arange(len(sequence_lengths))
    width = 0.35
    
    ax1.bar(x - width/2, standard_memory, width, label='Standard', alpha=0.8)
    ax1.bar(x + width/2, optimized_memory, width, label='Space-Time Optimized', alpha=0.8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Peak Memory (MB)')
    ax1.set_title('Memory Usage Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sequence_lengths)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Time comparison plot
    ax2.bar(x - width/2, standard_time, width, label='Standard', alpha=0.8)
    ax2.bar(x + width/2, optimized_time, width, label='Space-Time Optimized', alpha=0.8)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Inference Time (seconds)')
    ax2.set_title('Inference Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sequence_lengths)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_demo()
