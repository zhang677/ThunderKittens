import pandas as pd
import math
import matplotlib.pyplot as plt
import sys

config = {
    "l40s": {
        "l1": 3272, # GB / s
        "l2": 2180, # GB / s
        "dram": 864, # GB / s
    },
    "a100": {
        "l1": 19, # TB / s
        "l2": 7.2, # TB / s
        "dram": 1.55, # TB / s
    }
}

def get_util(df, tier):
    metric_mapping = {
        "l1": "L1/TEX Cache Throughput",
        "l2": "L2 Cache Throughput",
        "dram": "DRAM Throughput",
        "comp": "Compute (SM) Throughput"
    }
    metric_name = metric_mapping[tier]
    util_line = df[(df["Section Name"] == "GPU Speed Of Light Throughput") & (df["Metric Name"] == metric_name)]
    util_value = float(util_line["Metric Value"].values[0])
    return util_value * 0.01

def get_points(df, arch_config):
    peak_l1 = arch_config["l1"]
    util_comp = get_util(df, "comp")
    y = math.log10(util_comp)
    points = {}
    for tier in ["l1", "l2", "dram"]:
        util = get_util(df, tier)
        x = math.log10(peak_l1 / arch_config[tier]) + math.log10(util_comp / util)
        points[tier] = (x, y)
    return points

def plot_roofline(df, arch_config, output_file):
    points = get_points(df, arch_config)
    # Get min x and max x 
    x_min = min(points["l1"][0], points["l2"][0], points["dram"][0])
    x_max = max(points["l1"][0], points["l2"][0], points["dram"][0])
    ridge_x = {}
    peak_l1 = arch_config["l1"]
    for tier in ["l1", "l2", "dram"]:
        ridge_x[tier] = math.log10(peak_l1 / arch_config[tier])
    plot_min_y = min(points["l1"][1], x_min) - 0.2
    plot_min_x = plot_min_y
    plot_max_x = max(x_max, ridge_x["dram"]) + 0.5
    plt.figure(figsize=(12, 8))
    colors = {
        'l1': 'blue', 
        'l2': 'red', 
        'dram': 'green'
    }
    # Plot three ceilings first
    # The first ceiling ridge point is (0, 0) for(1, 100%)
    # A 45 deg line from (plot_min_x, -plot_min_x) to (0, 0)
    plt.plot([plot_min_x, 0], [plot_min_x, 0], color=colors['l1'], linestyle='--')
    # A 0 deg line from (0, 0) to (plot_max_x, 0)
    plt.plot([0, plot_max_x], [0, 0], color=colors['l1'], linestyle='--')
    # The second ceiling ridge point is (math.log10(peak_l1 / arch_config["l2"]), 0)
    plt.plot([ridge_x["l2"] + plot_min_x, ridge_x["l2"]], [plot_min_x, 0], color=colors['l2'], linestyle='--')
    plt.plot([ridge_x["l2"], plot_max_x], [0, 0], color=colors['l2'], linestyle='--')
    # The third ceiling ridge point is (math.log10(peak_l1 / arch_config["dram"]), 0)
    plt.plot([ridge_x["dram"] + plot_min_x, ridge_x["dram"]], [plot_min_x, 0], color=colors['dram'], linestyle='--')
    plt.plot([ridge_x["dram"], plot_max_x], [0, 0], color=colors['dram'], linestyle='--')
    # Plot the points
    for tier, point in points.items():
        print(f"{tier}: {point}")
        plt.scatter(point[0], point[1], color=colors[tier], label=tier)
    # Add labels
    plt.xlabel('Log10(Util_comp / Util_bw)')
    plt.ylabel('Log10(Util_comp)')
    plt.title('Utilization Roofline Model')

    plt.xlim(plot_min_x, plot_max_x)
    plt.ylim(plot_min_y, 0.2)
    plt.xticks(rotation=0)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def main(profile_file, arch, output_file):
    arch_config = config[arch]
    df = pd.read_csv(profile_file)
    plot_roofline(df, arch_config, output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python plot_roofline.py <profile_file> <arch> <output_file>")
        sys.exit(1)
    
    profile_file = sys.argv[1]
    arch = sys.argv[2]
    output_file = sys.argv[3]
    
    main(profile_file, arch, output_file)