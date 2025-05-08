import pandas as pd
import sys
import matplotlib.pyplot as plt

def get_latency_type(real_latency, latency_dict):
    candidates = {}
    for key, latency in latency_dict.items():
        if latency < real_latency:
            candidates[key] = latency
    if len(candidates) == 0:
        raise ValueError("No latency is less than real latency")
    if "Decoupled+spill" in candidates and "Coupled" in candidates:
        if candidates["Decoupled+spill"] == candidates["Coupled"]:
            del candidates["Decoupled+spill"]
    max_key = max(candidates, key=candidates.get)
    return candidates[max_key], max_key

colors = ['blue', 'red', 'green', 'purple']
markers = {"Coupled": 'x', "Decoupled": 's', 'Decoupled+spill': 'o'}

latency_record = "/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/plot_results/merge_spill_latency.csv"
df = pd.read_csv(latency_record)

m_list = [16, 32, 48, 64]
d_list = [64, 96, 128, 160]
mode = sys.argv[1]  # "d" or "m"
plt.figure(figsize=(10, 6))
plt.title(f"Latency for different {mode} values")
if mode == "m":
    outer_list = m_list
    inner_list = d_list
elif mode == "d":
    outer_list = d_list
    inner_list = m_list
else:
    raise ValueError("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
for sub_id, outer_value in enumerate(outer_list):
    real_latency_list = []
    latency_list = []
    latency_type_list = []
    batch_list = []
    err_list = []
    for inner_value in inner_list:
        if mode == "d":
            m = inner_value
            d = outer_value
        elif mode == "m":
            m = outer_value
            d = inner_value
        else:
            raise ValueError("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
        cur_df = df[df["shape"] == f"{m}x{d}"]
        real_latency = cur_df["min_latency"].values[0]
        simulated_latency = cur_df["simulated_latency"].values[0]
        long_simulated_latency = cur_df["inst_simulated_latency"].values[0]
        spill_simulated_latency = cur_df["spill_simulated_latency"].values[0]
        latency, latency_type = get_latency_type(real_latency, {
            "Decoupled": simulated_latency, 
            "Coupled": long_simulated_latency, 
            "Decoupled+spill": spill_simulated_latency
        })
        latency_list.append(latency)
        latency_type_list.append(latency_type)
        real_latency_list.append(real_latency)
        batch_list.append(int(cur_df["batch_min"].values[0]))
        err_list.append(f'{(latency - real_latency) / real_latency:.2%}')
    plt.plot(inner_list, latency_list, color=colors[sub_id], linestyle='--', label=f"{mode}={outer_value}, Err={err_list}")
    for inner_id in range(len(inner_list)):
        inner_value = inner_list[inner_id]
        latency_type = latency_type_list[inner_id]
        plt.plot(inner_value, latency_list[inner_id], color=colors[sub_id], marker=markers[latency_type])
    plt.plot(inner_list, real_latency_list, color=colors[sub_id], linestyle='-', marker='^', label=f"Real {mode}={outer_value}, batch={batch_list}")    

plt.ylim(1e5, 3e6)
plt.yscale('log')
if mode == "d":
    plt.xlabel("m")
elif mode == "m":
    plt.xlabel("d")
else:
    raise ValueError("Invalid mode. Use 'd' for latency vs d or 'm' for latency vs m.")
plt.ylabel("Latency (cycles)")
plt.legend()
plt.grid()
output_path = f"/scratch/zgh23/ThunderKittens/kernels/test_05_05_08_33/plot_results/merged_spill_latency_{mode}.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()