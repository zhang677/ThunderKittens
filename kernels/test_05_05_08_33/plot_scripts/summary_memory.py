import pandas as pd
import os

def get_value(df, name):
    return df.loc[df["Item"] == name, "Value"].iloc[0]

base_dir = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/profile_results_a100"
latency_csv = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/plot_results/merge_inst_latency.csv"
output_csv = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/plot_results/summary_memory.csv"
def merge_spill_stall(latency_csv, output_csv):
    records = []
    latency_df = pd.read_csv(latency_csv)
    for i in range(len(latency_df)):
        m, d = latency_df["shape"][i].split('x')
        b = str(latency_df["batch_min"][i])
        shape = 'x'.join([b, '108', m, '6336', d])
        memory_record = os.path.join(base_dir, f'memory_{shape}.csv')
        df = pd.read_csv(memory_record)
        spill = float(df.loc[df["Item"] == "L1_to_Local_traffic", "Value"].iloc[0]) / float(df.loc[df["Item"] == "DRAM_to_L2_traffic", "Value"].iloc[0])
        stalls = {
            "Stall_wait": get_value(df, "Stall_wait"),
            "Stall_long_scoreboard": get_value(df, "Stall_long_scoreboard"),
            "Stall_short_scoreboard": get_value(df, "Stall_short_scoreboard"),
        }
        stall_reason = max(stalls, key=stalls.get)
        stall_value = stalls[stall_reason]
        warp_cycles = float(get_value(df, "Warp_cycles_per_exec"))
        records.append({
            "shape": f"{m}x{d}",
            "spill": spill,
            "stall_reason": stall_reason,
            "stall_value": stall_value,
            "warp_cycles": warp_cycles
        })
    # Create a DataFrame from the records
    records_df = pd.DataFrame(records)
    # Write to output CSV
    records_df.to_csv(output_csv, index=False)
    print(f"Memory records merged into {output_csv}")

if __name__ == "__main__":
    merge_spill_stall(latency_csv, output_csv)

    

