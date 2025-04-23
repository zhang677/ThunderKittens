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

def plot_error_heatmap(output_file):
    plt.figure(figsize=(12, 8))

    base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results"
    error_matrix = np.zeros((4, 4))
    for i, m in enumerate([16, 32, 48, 64]):
        for j, d in enumerate([64, 96, 128, 160]):
            filename = f"{base_dir}/output_17x8x{m}x{d}.csv"
            run_last_line = load_and_process_file(filename)
            sim_file = filename.replace('output', 'simulated')
            sim_last_line = load_and_process_file(sim_file)
            # Calculate percentage error
            error = 100 * (sim_last_line['latency'] - run_last_line['latency']) / run_last_line['latency']
            error_matrix[i, j] = error
    
    # Create a heatmap
    blues_cmap = plt.cm.Blues
    
    # Create a heatmap with blue color scheme
    im = plt.imshow(np.abs(error_matrix), cmap=blues_cmap, interpolation='nearest')
    
    # Add colorbar
    cbar = plt.colorbar(im, label='Relative Error (%)')
    
    # Add percentage text on each cell
    for i in range(4):
        for j in range(4):
            text_color = 'white' if abs(error_matrix[i, j]) > np.max(error_matrix) * 0.7 else 'black'
            plt.text(j, i, f"{error_matrix[i, j]:.1f}%", 
                     ha="center", va="center", color=text_color,
                     fontsize=11, fontweight='bold')
    
    plt.xticks(np.arange(4), ['64', '96', '128', '160'])
    plt.yticks(np.arange(4), ['16', '32', '48', '64'])
    plt.xlabel('D', fontsize=12, fontweight='bold')
    plt.ylabel('M', fontsize=12, fontweight='bold')
    plt.title('Relative Error Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_rel_error_heatmap.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    plot_error_heatmap(output_file)
    print(f"Plot saved as {output_file}")