import pandas as pd
import os
import sys

def extract_h(shape_str):
    """Extract the H dimension (1st value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[1])
    return 0

def extract_n(shape_str):
    """Extract the N dimension (4th value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[3])
    return 0

def extract_m_d(shape_str):
    # Extract the shape from the string
    shape = shape_str.split("x")
    int_shape = tuple(map(int, shape))
    if len(shape) == 5:
        return (int_shape[2], int_shape[4])
    elif len(shape) == 4:
        return (int_shape[1], int_shape[3])
    else:
        raise ValueError(f"Invalid shape format: {shape_str}")
    
def get_max_latency(filename):
    """Load a shape-latency CSV file and extract the N dimension"""
    df = pd.read_csv(filename)
    
    # Extract N dimension from shape string
    df['N'] = df['shape'].apply(extract_n)
    
    # Sort by N dimension
    df = df.sort_values('N')
    
    # Get the last line (largest N)
    last_line = df.iloc[-1]
    
    return last_line['latency']

def get_item(df, label, shape):
    return float(df.loc[df["shape"] == shape, label].iloc[0])

base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_13_28/profile_results_a100"
output_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_13_28/plot_results" 
sim_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/profile_results_a100"
inst_sim_dir = "/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_a100"

# Enumerate over all the files start with "output_" under base_dir
with open(os.path.join(output_dir, "merge_spill_latency.csv"), "w") as outf:
    outf.write("shape,batch_min,min_latency,simulated_latency,inst_simulated_latency,spill_simulated_latency,inst_spill_simulated_latency\n")
    merge_warp_latency_file = os.path.join(output_dir, "merge_warp_latency.csv")
    merge_warp_latency_df = pd.read_csv(merge_warp_latency_file)
    summary_memory_file = os.path.join(output_dir, "summary_memory.csv")
    summary_memory_df = pd.read_csv(summary_memory_file)
    for i in range(len(merge_warp_latency_df)):
        shape = merge_warp_latency_df["shape"][i]
        m, d = extract_m_d(shape)
        simulated_file = os.path.join(sim_dir, f"simulated_{m}x{d}.csv")
        simulated_latency = get_max_latency(simulated_file)
        long_simulated_file = os.path.join(inst_sim_dir, f"simulated_13x8x{m}x{d}.csv")
        long_simulated_latency = get_max_latency(long_simulated_file)
        spill_factor = get_item(summary_memory_df, "spill", shape)
        spill_simulated_latency = simulated_latency * (1 + spill_factor)
        inst_spill_simulated_latency = long_simulated_latency * (1 + spill_factor)
        outf.write(f"{m}x{d},{extract_h(shape)},{get_item(merge_warp_latency_df, "latency", shape)},{simulated_latency},{long_simulated_latency},{spill_simulated_latency},{inst_spill_simulated_latency}\n")
        
