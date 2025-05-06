import pandas as pd
import os
import sys

def extract_n(shape_str):
    """Extract the N dimension (4th value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[3])
    return 0

def extract_m_d(shape_str):
    # Extract the shape from the string
    shape = shape_str.split("x")
    if len(shape) != 5:
        raise ValueError(f"Invalid shape format: {shape_str}")
    int_shape = tuple(map(int, shape))
    return (int_shape[2], int_shape[4])

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

base_dir = sys.argv[1]
output_dir = sys.argv[2]
sim_dir = "/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_a100"

# Enumerate over all the files start with "output_" under base_dir
with open(os.path.join(output_dir, "merge_min_latency.csv"), "w") as outf:
    outf.write("shape,min_latency,simulated_latency,long_simulated_latency\n")
    for file in os.listdir(base_dir):
        if file.startswith("output_"):
            input_file = os.path.join(base_dir, file)
            with open(input_file, "r") as f:
                df = pd.read_csv(f)
                shapes = df["shape"]
                latencies = df["latency_per_batch"]
                min_latency = min(latencies)
            m, d = extract_m_d(shapes[0])
            simulated_file = os.path.join(base_dir, f"simulated_{m}x{d}.csv")
            simulated_latency = get_max_latency(simulated_file)
            long_simulated_file = os.path.join(sim_dir, f"simulated_13x8x{m}x{d}.csv")
            long_simulated_latency = get_max_latency(long_simulated_file)
            outf.write(f"{m}x{d},{min_latency},{simulated_latency},{long_simulated_latency}\n")
        
