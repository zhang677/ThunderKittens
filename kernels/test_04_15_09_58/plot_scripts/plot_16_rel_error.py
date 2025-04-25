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
    
    return df

def plot_latency_curves(files, output_file="latency_plot.png"):
    """Plot latency vs N curves for multiple files"""
    plt.figure(figsize=(12, 8))
    
    # Define colors and markers for different curves
    # colors = ['blue', 'red', 'green', 'purple']
    # markers = ['o', 's', '^', 'x']
    # Extend to support 16 curves
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray',
             'olive', 'cyan', 'magenta', 'yellow', 'teal', 'navy', 'maroon', 'lime']
    markers = ['o', 's', '^', 'x', 'D', 'H', 'v', '<', '>', 'p', '*', 'X', '|', '_', '+', '.']
    
    # min_latency = float('inf')
    # max_latency = 0
    
    for i, file in enumerate(files):
        if i >= 16:  # Limit to 16 files
            print(f"Warning: Only processing the first 16 files. Skipping {file}")
            continue
            
        if not os.path.exists(file):
            print(f"Error: File {file} does not exist")
            continue
            
        # Get the filename without extension for the legend
        label = os.path.splitext(os.path.basename(file))[0].split('_')[1]
        
        # Load and process data
        try:
            df = load_and_process_file(file)
            
            sim_file = file.replace('output', 'simulated')
            df_sim = load_and_process_file(sim_file)
            # Plot the curve
            # Plot the curve - calculate percentage error
            plt.plot(df['N'], 100 * (df_sim['latency'] - df['latency']) / df['latency'], 
                     marker=markers[i], 
                     color=colors[i], 
                     linestyle='-', 
                     linewidth=2, 
                     markersize=8,
                     label=label)
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Add labels and legend with percentage formatting
    plt.xlabel('N Dimension', fontsize=14)
    plt.ylabel('Error (%)', fontsize=14)  # Updated label
    plt.title('(Sim - Real) / Real vs N Dimension', fontsize=16)  # Updated title
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Format y-axis as percentage
    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved as {output_file}")


def main(arch, output_file):
    """Main function to parse command line arguments and plot data"""
    if arch == "l40s":
        shape = "17x8"
    elif arch == "a100":
        shape = "13x8"
    else:
        print("Unsupported architecture. Please use 'l40s' or 'a100'.")
        sys.exit(1)
    base_dir = f"/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_{arch}"
    files = []
    for m in [16, 32, 48, 64]:
        for d in [64, 96, 128, 160]:
            filename = f"{base_dir}/output_{shape}x{m}x{d}.csv"
            files.append(filename)
    
    # Plot the latency curves
    plot_latency_curves(files, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_16_rel_error.py <arch> <output_file>")
        sys.exit(1)
    arch = sys.argv[1]
    output_file = sys.argv[2]
    main(arch, output_file)