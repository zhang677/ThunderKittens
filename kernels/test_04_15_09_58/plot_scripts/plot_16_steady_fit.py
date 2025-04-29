import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import sys
import os

def extract_n(shape_str):
    """Extract the N dimension (4th value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[3])
    return 0

def load_and_process_file(filename):
    """Load a shape-latency CSV file and extract the N dimension"""
    df = pd.read_csv(filename)
    
    # Extract N dimension from shape string
    df['N'] = df['shape'].apply(extract_n)
    
    # Sort by N dimension
    df = df.sort_values('N')
    
    # Get the last line (largest N)
    last_line = df.iloc[-1]
    
    return last_line

def calculate_roofline(last_line, bandwidth, mm_thpt, vec_thpt):
    shape = [int(x) for x in last_line['shape'].split('x')]
    b, h, m, n, d = shape
    total_bytes = b * h * (2 * m * d * 2 + 2 * n * d * 2)
    compute_latency = b * h * (2 * m * n * d * 2 / mm_thpt + (2 * m * 16 + m * d) * n / 16 / vec_thpt)
    memory_latency = total_bytes / bandwidth
    return max(memory_latency, compute_latency)

def plot_latency_d(arch, output_file):
    plt.figure(figsize=(12, 8))

    if arch == "a100":
        base_shape = "13x8"
        freq = 1.41 * 1e9
        bandwidth = 1.5 * 1e12 / freq
        mm_thpt = 312 * 1e12 / freq
        vec_thpt = 39 * 1e12 / freq
    elif arch == "l40s":
        base_shape = "17x8"
        freq = 1.06 * 1e9
        bandwidth = 864 * 1e9 / freq
        mm_thpt = 362 * 1e12 / freq
        vec_thpt = 91.6 * 1e12 / freq
    else:
        raise ValueError("Unsupported architecture. Use 'a100' or 'l40s'.") 

    base_dir = f"/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_{arch}"
    
    # Define m and d values
    m_values = [16, 32, 48, 64]
    d_values = [64, 96, 128, 160]
    
    # Define colors and markers for different curves
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    
    # Create plot for latency vs. d for each m
    plt.figure(figsize=(10, 6))
    
    for i, m in enumerate(m_values):
        real_latencies = []
        slopes = []
        intercepts = []
        roofline_latencies = []
        for d in d_values:
            filename = f"{base_dir}/output_{base_shape}x{m}x{d}.csv"
            run_last_line = load_and_process_file(filename)
            
            real_latencies.append(run_last_line['latency'])
            roofline_latency = calculate_roofline(run_last_line, bandwidth, mm_thpt, vec_thpt)
            roofline_latencies.append(roofline_latency)
        
        for j in range(len(real_latencies) - 1):
            slope = (real_latencies[j + 1] - real_latencies[j]) / (d_values[j + 1] - d_values[j])
            intercept = real_latencies[j] - slope * d_values[j]
            slopes.append(slope)
            intercepts.append(intercept)

        # Roofline slope and intercept
        roofline_slope = (roofline_latencies[-1] - roofline_latencies[0]) / (d_values[-1] - d_values[0])
        roofline_intercept = roofline_latencies[0] - roofline_slope * d_values[0]  
        # Plot real data with solid lines (-)
        plt.plot(d_values, real_latencies, color=colors[i], marker=markers[i], 
                 linestyle='-', label=f'Real m={m}; Roofline: k={roofline_slope:.2f}, b={roofline_intercept:.2f}')
        
        # Display slope and intercept on top of each line
        for j in range(len(slopes)):
            plt.text((d_values[j] + d_values[j+1]) / 2, (real_latencies[j] + real_latencies[j+1]) / 2, f"k: {slopes[j]:.2f}  b: {intercepts[j]:.2f}", 
                     fontsize=8, color=colors[i], ha='center', va='bottom')
        
    plt.xlabel('D', fontsize=12)
    plt.ylabel('Latency (cycles)', fontsize=12)
    plt.title('Latency vs. d for Different Values of m', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def plot_latency_m(arch, output_file):
    plt.figure(figsize=(12, 8))
    if arch == "a100":
        base_shape = "13x8"
    elif arch == "l40s":
        base_shape = "17x8"
    else:
        raise ValueError("Unsupported architecture. Use 'a100' or 'l40s'.") 
    base_dir = f"/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_{arch}"
    
    # Define m and d values
    m_values = [16, 32, 48, 64]
    d_values = [64, 96, 128, 160]
    
    # Define colors and markers for different curves
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    
    # Create plot for latency vs. d for each m
    plt.figure(figsize=(10, 6))
    
    for i, d in enumerate(d_values):
        real_latencies = []
        slopes = []
        intercepts = []
        for m in m_values:
            filename = f"{base_dir}/output_{base_shape}x{m}x{d}.csv"
            run_last_line = load_and_process_file(filename)
            
            real_latencies.append(run_last_line['latency'])
        
        for j in range(len(real_latencies) - 1):
            slope = (real_latencies[j + 1] - real_latencies[j]) / (m_values[j + 1] - m_values[j])
            intercept = real_latencies[j] - slope * m_values[j]
            slopes.append(slope)
            intercepts.append(intercept)
            
        # Plot real data with solid lines (-)
        plt.plot(m_values, real_latencies, color=colors[i], marker=markers[i], 
                 linestyle='-', label=f'Real m={m}')
        
        # Display slope and intercept on top of each line
        for j in range(len(slopes)):
            plt.text((m_values[j] + m_values[j+1]) / 2, (real_latencies[j] + real_latencies[j+1]) / 2, f"k: {slopes[j]:.2f}\nb: {intercepts[j]:.2f}", 
                     fontsize=8, color=colors[i], ha='center', va='bottom')
    
    plt.xlabel('M', fontsize=12)
    plt.ylabel('Latency (cycles)', fontsize=12)
    plt.title('Latency vs. m for Different Values of d', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python plot_16_steady.py <{d,m}> arch <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[3]
    arch = sys.argv[2]
    mode = sys.argv[1]
    if mode == "d":
        plot_latency_d(arch, output_file)
    elif mode == "m":
        plot_latency_m(arch, output_file)
    else:
        print("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
        sys.exit(1)
    print(f"Plot saved as {output_file}")