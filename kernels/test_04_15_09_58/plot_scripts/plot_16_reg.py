import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import sys
import os

def extract_reg(filename):
    df = pd.read_csv(filename)
    df_line = df[(df["Section Name"] == "Launch Statistics") & (df["Metric Name"] ==  "Registers Per Thread")]
    return int(df_line["Metric Value"].values[0])

def plot_reg_heatmap(output_file):
    plt.figure(figsize=(12, 8))

    base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results"
    reg_matrix = np.zeros((4, 4), dtype=int)
    for i, m in enumerate([16, 32, 48, 64]):
        for j, d in enumerate([64, 96, 128, 160]):
            filename = f"{base_dir}/short_17x8x{m}x6336x{d}.csv"
            num_reg = extract_reg(filename)
            # Calculate percentage error
            reg_matrix[i, j] = num_reg
    # Create a heatmap
    blues_cmap = plt.cm.Blues
    
    # Create a heatmap with blue color scheme
    im = plt.imshow(reg_matrix, cmap=blues_cmap, interpolation='nearest')

    cbar = plt.colorbar(im, label='Num regs')

    for i in range(4):
        for j in range(4):
            text_color = 'white' if reg_matrix[i, j] > 178 else 'black'
            plt.text(j, i, f"{reg_matrix[i, j]}", 
                     ha="center", va="center", color=text_color,
                     fontsize=11, fontweight='bold')

    plt.xticks(np.arange(4), ['64', '96', '128', '160'])
    plt.yticks(np.arange(4), ['16', '32', '48', '64'])
    plt.xlabel('D', fontsize=12, fontweight='bold')
    plt.ylabel('M', fontsize=12, fontweight='bold')
    plt.title('Register Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_16_reg.py <output_file>")
        sys.exit(1)
    
    output_file = sys.argv[1]
    plot_reg_heatmap(output_file)
    print(f"Plot saved as {output_file}")