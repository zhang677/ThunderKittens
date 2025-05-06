import sys
import pandas as pd
import os

def extract_batch(shape_str):
    # Extract the shape from the string
    shape = shape_str.split("x")
    if len(shape) != 5:
        raise ValueError(f"Invalid shape format: {shape_str}")
    return tuple(map(int, shape))[0]

base_dir = sys.argv[1]

# Enumerate over all the files start with "output_" under base_dir
for file in os.listdir(base_dir):
    if file.startswith("output_"):
        input_file = os.path.join(base_dir, file)
        output_file = os.path.join(base_dir, "temp_" + file)
        with open(input_file, "r") as f:
            df = pd.read_csv(f)
            shapes = df["shape"]
            latencies = df["latency"]
            batches = [extract_batch(shape) for shape in shapes]
            latency_per_batch = [latency / batch for latency, batch in zip(latencies, batches)]
            df["latency_per_batch"] = latency_per_batch
            df.to_csv(output_file, index=False)
            os.remove(input_file)
            os.rename(output_file, input_file)
            print(f"Updated {input_file} with latency per batch.")