import os
import pandas as pd

def get_value(df, name):
    return df.loc[df["Item"] == name, "Value"].iloc[0]

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

output_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_13_28/plot_results" 
base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_13_28/profile_results_a100"

with open(os.path.join(output_dir, "merge_warp_latency.csv"), "w") as outf:
    outf.write("shape,num_warps,latency\n")
    base_dir_list = os.listdir(base_dir)
    base_dir_list = [file for file in base_dir_list if file.startswith("memory_")]
    # Sort the list of files based on mxd
    base_dir_list.sort(key=lambda x: extract_m_d(x.split("_")[1].split(".")[0]))
    for file in base_dir_list:
        input_file = os.path.join(base_dir, file)
        with open(input_file, "r") as f:
            df = pd.read_csv(f)
            shape = file.split("_")[1].split(".")[0]
            m, d = extract_m_d(shape)
            num_warps = get_value(df, "Num_warps")
            latency = get_value(df, "cycles") / num_warps
            outf.write(f"{shape},{num_warps},{latency}\n")