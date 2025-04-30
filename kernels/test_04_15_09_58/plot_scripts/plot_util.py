import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_util(base_dir, mode):
    if mode == "a100":
        b = 13
        h = 8
    elif mode == "l40s":
        b = 17
        h = 8
    else:
        raise ValueError("Unsupported architecture. Use 'a100' or 'l40s'.")
    l1_util_list = []
    l2_util_list = []
    dram_util_list = []
    comp_util_list = []
    x_axis_list = []
    for m in [16, 32, 48, 64]:
        for d in [64, 96, 128, 160]:
            for n in [256, 512, 1536, 2560, 3584, 4288, 5312, 6336]:
                filename = f"{base_dir}/short_{b}x{h}x{m}x{n}x{d}.csv"
                df = pd.read_csv(filename)
                l1_util_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "L1/TEX Cache Throughput")]
                l2_util_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "L2 Cache Throughput")]
                dram_util_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "DRAM Throughput")]
                comp_util_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] ==  "Compute (SM) Throughput")]
                l1_util = float(l1_util_line["Metric Value"].values[0]) * 0.01
                l2_util = float(l2_util_line["Metric Value"].values[0]) * 0.01
                dram_util = float(dram_util_line["Metric Value"].values[0]) * 0.01
                comp_util = float(comp_util_line["Metric Value"].values[0]) * 0.01
                l1_util_list.append(l1_util)
                l2_util_list.append(l2_util)
                dram_util_list.append(dram_util)
                comp_util_list.append(comp_util)
                x_axis_list.append(f"{m}x{n}x{d}")
    
    # Plotting
    plt.figure(figsize=(20, 8))
    colors = ['blue', 'red', 'green', 'purple']
    markers = ['o', 's', '^', 'x']
    # Plot every 8 points
    for i in range(0, len(x_axis_list), 8):
        if i == 0:
            labels = ['L1 Utilization', 'L2 Utilization', 'DRAM Utilization', 'Compute Utilization']
        else:
            labels = [None, None, None, None]
        plt.plot(x_axis_list[i:i+8], l1_util_list[i:i+8], label=labels[0], marker=markers[0], color=colors[0])
        plt.plot(x_axis_list[i:i+8], l2_util_list[i:i+8], label=labels[1], marker=markers[1], color=colors[1])
        plt.plot(x_axis_list[i:i+8], dram_util_list[i:i+8], label=labels[2], marker=markers[2], color=colors[2])
        plt.plot(x_axis_list[i:i+8], comp_util_list[i:i+8], label=labels[3], marker=markers[3], color=colors[3])
    # Only legend once
    plt.xlabel('Shape (m x n x d)')
    plt.ylabel('Utilization (%)')
    plt.title(f'Utilization vs Shape for {mode}')
    plt.xticks(rotation=90, fontsize=6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{base_dir}/util_plot_{mode}.png")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_util.py <base_dir> <mode>")
        sys.exit(1)
    
    mode = sys.argv[2]

    plot_util(sys.argv[1], mode)