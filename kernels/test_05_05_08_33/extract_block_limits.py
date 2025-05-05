import ncu_report
import sys

metrics = {
    "Registers": "launch__occupancy_limit_registers",
    "Shared_Mem": "launch__occupancy_limit_shared_mem",
    "Warps": "launch__occupancy_limit_warps",
    "SM": "launch__occupancy_limit_blocks",
}

def extract_values(ncu_report_file, output_csv):
    results = {}
    my_context = ncu_report.load_report(ncu_report_file)
    my_range = my_context.range_by_idx(0)
    for j in range(my_range.num_actions()):
        my_action = my_range.action_by_idx(j)
        kernel_name = my_action.name()
        for (nick_name, metric_name) in metrics.items():
            if my_action.metric_by_name(metric_name) is None:
                raise ValueError(f"Metric {metric_name} not found in action {kernel_name}")
            else:
                results[nick_name] = my_action.metric_by_name(metric_name).as_uint64()
    with open(output_csv, "w") as f:
        f.write("Item,Value\n")
        for (nick_name, metric_name) in metrics.items():
            f.write(f"{nick_name},{results[nick_name]}\n")
        f.write("Minimum,{}\n".format(min(results.values())))

if __name__ == "__main__":
    ncu_report_file = sys.argv[1]
    output_csv = sys.argv[2]
    extract_values(ncu_report_file, output_csv)