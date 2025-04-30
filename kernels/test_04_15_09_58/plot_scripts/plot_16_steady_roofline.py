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

def extract_total_bytes(df):
    thpt_line = df[(df["Section Name"] == "Memory Workload Analysis") & (df["Metric Name"] ==  "Memory Throughput")]
    thpt_unit = thpt_line["Metric Unit"].values[0]
    thpt_value = float(thpt_line["Metric Value"].values[0])
    if thpt_unit == "Gbyte/s":
        thpt_value *= 1e9
    elif thpt_unit == "Mbyte/s":
        thpt_value *= 1e6
    else:
        raise ValueError(f"Unknown unit: {thpt_unit}")
    
    time_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "Duration")]
    time_unit = time_line["Metric Unit"].values[0]
    time_value = float(time_line["Metric Value"].values[0])
    if time_unit == "ms":
        time_value *= 1e-3
    elif time_unit == "us":
        time_value *= 1e-6
    elif time_unit == "ns":
        time_value *= 1e-9
    else:
        raise ValueError(f"Unknown unit: {time_unit}")
    
    return thpt_value * time_value

def calculate_roofline(last_line, base_dir, bandwidth, mm_thpt, vec_thpt):
    profile_file = os.path.join(base_dir, f"short_{last_line['shape']}.csv")
    df = pd.read_csv(profile_file)
    shape = [int(x) for x in last_line['shape'].split('x')]
    total_bytes = extract_total_bytes(df)
    b, h, m, n, d = shape
    minimal_bytes = b * h * (2 * m * d * 2 + 2 * n * d * 2)
    compute_latency = b * h * (2 * m * n * d * 2 / mm_thpt + (2 * m * 16 + m * d) * n / 16 / vec_thpt)
    memory_latency = total_bytes / bandwidth
    rel_err = (total_bytes - minimal_bytes) / total_bytes
    return max(memory_latency, compute_latency), rel_err

def get_memory_utilization(last_line, base_dir):
    profile_file = os.path.join(base_dir, f"short_{last_line['shape']}.csv")
    df = pd.read_csv(profile_file)
    thpt_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "DRAM Throughput")]
    return float(thpt_line["Metric Value"].values[0]) * 0.01

def plot_latency(arch, output_file, mode):
    plt.figure(figsize=(12, 8))

    if arch == "a100":
        base_shape = "13x8"
        freq = 1.10 * 1e9
        bandwidth = 1.55 * 1e12 / freq
        mm_thpt = 312 * 1e12 / freq
        vec_thpt = 39 * 1e12 / freq
    elif arch == "l40s":
        base_shape = "17x8"
        freq = 1.065 * 1e9
        bandwidth = 864 * 1e9 / freq
        mm_thpt = 362 * 1e12 / freq
        vec_thpt = 91.6 * 1e12 / freq
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
    last_lines = []
    for i, out_v in enumerate(out_values):
        real_latencies = []
        sim_latencies = []
        roofline_latencies = []
        rel_errors = []
        roofline_ratio = []
        real_ratio = []
        simulation_ratio = []
        
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
            last_lines.append(run_last_line)
            
            real_latencies.append(run_last_line['latency'])
            sim_latencies.append(sim_last_line['latency'])
            roofline_latency, rel_err = calculate_roofline(sim_last_line, base_dir, bandwidth, mm_thpt, vec_thpt)
            roofline_latencies.append(roofline_latency)
            rel_errors.append(rel_err)
            roofline_ratio.append(roofline_latency / real_latencies[-1])
            real_ratio.append(get_memory_utilization(run_last_line, base_dir))
            simulation_ratio.append(sim_latencies[-1] / real_latencies[-1])
        
        bytes_error_str = ", ".join([f"{e:.2f}" for e in rel_errors])
        plt.plot(in_values, roofline_latencies, color=colors[i], linestyle=':',
                label=f'Roofline {mode}={out_v}; Bytes Err: {bytes_error_str}', linewidth=2)
        # Plot real data with solid lines (-)
        plt.plot(in_values, real_latencies, color=colors[i], marker=markers[i], 
                 linestyle='-')
        
        # Plot simulated data with dashed lines (--)
        plt.plot(in_values, sim_latencies, color=colors[i], marker=markers[i], 
                 linestyle='--')

        for j in range(len(real_latencies)):
            plt.text(in_values[j], real_latencies[j], f"M: {100 * real_ratio[j]:.1f}% R: {100 * roofline_ratio[j]:.1f}% S: {100 * simulation_ratio[j]:.1f}%", 
                     fontsize=8, color=colors[i], ha='center', va='bottom')

    plt.yscale('log')
    plt.xlabel(mode, fontsize=12)
    plt.ylabel('Latency (cycles)', fontsize=12)
    plt.title(f'{arch} Latency vs. {mode}', fontsize=14)
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