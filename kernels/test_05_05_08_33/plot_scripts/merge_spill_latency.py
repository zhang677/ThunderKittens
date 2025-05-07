import pandas as pd
import os
import sys

def extract_b(shape_str):
    """Extract the B dimension (1st value) from the shape string"""
    parts = shape_str.split('x')
    if len(parts) >= 5:
        return int(parts[0])
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

def get_spill(df, shape):
    return float(df.loc[df["shape"] == shape, "spill"].iloc[0])

base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/profile_results_a100"
output_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/plot_results" 
sim_dir = "/scratch/zgh23/ThunderKittens/kernels/test_04_15_09_58/profile_results_a100"

# Enumerate over all the files start with "output_" under base_dir
with open(os.path.join(output_dir, "merge_spill_latency.csv"), "w") as outf:
    outf.write("shape,batch_min,min_latency,batch_occupancy,occupancy_latency,simulated_latency,inst_simulated_latency,spill_simulated_latency,inst_spill_simulated_latency\n")
    base_dir_list = os.listdir(base_dir)
    # Filter the list to only include files that start with "output_"
    base_dir_list = [file for file in base_dir_list if file.startswith("output_")]
    # Sort the list of files based on mxd
    base_dir_list.sort(key=lambda x: extract_m_d(x.split("_")[1].split(".")[0]))
    summary_memory_file = os.path.join(output_dir, "summary_memory.csv")
    summary_memory_df = pd.read_csv(summary_memory_file)
    for file in base_dir_list:
        if file.startswith("output_"):
            input_file = os.path.join(base_dir, file)
            with open(input_file, "r") as f:
                df = pd.read_csv(f)
                shapes = df["shape"]
                latencies = df["latency_per_batch"]
                # min_latency = min(latencies)
                # Get the index of the minimum latency
                min_index = latencies.idxmin()
                # Get the shape corresponding to the minimum latency
                min_shape = shapes[min_index]
                min_latency = latencies[min_index]
                batch_calc = extract_b(df.iloc[-3]['shape'])
                batch_min = extract_b(min_shape)
                calc_latency = df.iloc[-3]['latency_per_batch']
                

            m, d = extract_m_d(shapes[0])
            simulated_file = os.path.join(base_dir, f"simulated_{m}x{d}.csv")
            simulated_latency = get_max_latency(simulated_file)
            long_simulated_file = os.path.join(sim_dir, f"simulated_13x8x{m}x{d}.csv")
            long_simulated_latency = get_max_latency(long_simulated_file)
            spill_factor = get_spill(summary_memory_df, f"{m}x{d}")
            spill_simulated_latency = simulated_latency * (1 + spill_factor)
            inst_spill_simulated_latency = long_simulated_latency * (1 + spill_factor)
            outf.write(f"{m}x{d},{batch_min},{min_latency},{batch_calc},{calc_latency},{simulated_latency},{long_simulated_latency},{spill_simulated_latency},{inst_spill_simulated_latency}\n")
        
