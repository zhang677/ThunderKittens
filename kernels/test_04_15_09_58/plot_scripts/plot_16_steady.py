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

def get_memory_utilization(last_line, base_dir):
    profile_file = os.path.join(base_dir, f"short_{last_line['shape']}.csv")
    df = pd.read_csv(profile_file)
    thpt_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "DRAM Throughput")]
    return float(thpt_line["Metric Value"].values[0]) * 0.01

def plot_latency(arch, output_file, mode):
    plt.figure(figsize=(12, 8))

    if arch == "a100":
        base_shape = "13x8"
    elif arch == "l40s":
        base_shape = "17x8"
    else:
        raise ValueError("Unsupported architecture. Use 'a100' or 'l40s'.") 

    base_dir = f"/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_{arch}"
    
    # Define m and d values
    if mode == "d":
        out_values = [16, 32, 48, 64]
        in_values = [64, 96, 128, 160]
    elif mode == "m":
        out_values = [64, 96, 128, 160]
        in_values = [16, 32, 48, 64]
    else:
        raise ValueError("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
    
    # Define colors and markers for different curves
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    
    # Create plot for latency vs. d for each m
    plt.figure(figsize=(10, 6))
    
    for i, out_v in enumerate(out_values):
        real_latencies = []
        sim_latencies = []
        simulation_ratio = []
        real_ratio = []
        for in_v in in_values:
            if mode == "d":
                d = in_v
                m = out_v
            elif mode == "m":
                d = out_v
                m = in_v
            else:
                raise ValueError("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
            filename = f"{base_dir}/output_{base_shape}x{m}x{d}.csv"
            run_last_line = load_and_process_file(filename)
            sim_file = filename.replace('output', 'simulated')
            sim_last_line = load_and_process_file(sim_file)
            
            real_latencies.append(run_last_line['latency'])
            sim_latencies.append(sim_last_line['latency'])
            real_ratio.append(get_memory_utilization(run_last_line, base_dir))
            simulation_ratio.append(sim_latencies[-1] / real_latencies[-1])
            
        # Plot real data with solid lines (-)
        plt.plot(in_values, real_latencies, color=colors[i], marker=markers[i], 
                 linestyle='-', label=f'Real m={m}')
        
        # Plot simulated data with dashed lines (--)
        plt.plot(in_values, sim_latencies, color=colors[i], marker=markers[i], 
                 linestyle='--', label=f'Sim m={m}')
        for j in range(len(real_latencies)):
            plt.text(in_values[j], real_latencies[j], f"M: {real_ratio[j]:.3f} S: {simulation_ratio[j]:.3f}", 
                     fontsize=8, color=colors[i], ha='center', va='bottom')
    plt.yscale('log')
    plt.xlabel(mode, fontsize=12)
    plt.ylabel('Log Latency (cycles)', fontsize=12)
    plt.title(f'Latency vs. {mode}', fontsize=14)
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
    plot_latency(arch, output_file, mode)
    print(f"Plot saved as {output_file}")