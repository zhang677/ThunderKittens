import ncu_report
import sys

metrics = {
    "Normal_shmem_ld": "sass__inst_executed_shared_loads", # Number of shared memory load instructions executed other than LDSM
    "Atom_shmem_ld": "smsp__inst_executed_op_shared_atom.sum",
    "LDSM": "smsp__inst_executed_op_ldsm.sum", # LDSM: shmem to register using ldmatrix
    "LDGSTS": "smsp__inst_executed_op_ldgsts.sum", # LDGSTS requests
    "Normal_global_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", # # of requests sent to T-Stage for global loads
    "Atom_global_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "LDGSTS_global_ld": "sm__sass_l1tex_t_requests_pipe_lsu_mem_global_op_ldgsts.sum",
    "LDGSTS_traffic": "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum", # LDGSTS traffic (L2 to shmem)
    "L2_to_L1_traffic": "l1tex__m_xbar2l1tex_read_bytes.sum", # # of bytes read from L2 into L1TEX M-Stage
    "DRAM_to_L2_traffic": "dram__bytes_read.sum", # # of bytes read from DRAM
    "L2_throughput": "l1tex__m_xbar2l1tex_read_bytes.sum.per_second", # L2 measured throughput
    "DRAM_throughput": "dram__bytes_read.sum.per_second", # DRAM measured throughput
    "Normal_global_st": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "Atom_global_st": "l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum",
    "Red_global_st": "l1tex__t_requests_pipe_lsu_mem_global_op_red.sum",
    "Normal_shmem_st": "sass__inst_executed_shared_stores",
    "Atom_shmem_st": "smsp__inst_executed_op_shared_atom.sum",
    "Duration": "gpu__time_duration.sum",
    "Cycles": "smsp__cycles_elapsed.max", 
    "L1_util": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "L2_util": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "DRAM_util": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}

def extract_values(ncu_report_file, problem_shape, output_csv):
    assert problem_shape in ncu_report_file, f"Shape {problem_shape} does not match {ncu_report_file}"
    b, h, n, m, d = tuple(map(int, problem_shape.split("x")))
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
                results[nick_name] = my_action.metric_by_name(metric_name).as_double()

    # Normal global traffic also goes through L1
    normal_global_traffic = (results["Normal_global_ld"] + results["Atom_global_ld"] - results["LDGSTS_global_ld"]) * 16
    L1_to_Reg_traffic = results["Normal_shmem_ld"] * 16 + results["LDSM"] * 512 + results["Atom_shmem_ld"] * 16 + normal_global_traffic
    L1_to_Reg_traffic_calc = b * h * (m + n * 2) * d * 2
    global_load_calc = b * h * (m + n * 2) * d * 2
    global_load_from_req = results["LDGSTS"] * 512 + normal_global_traffic
    measured_traffic_delta = results["DRAM_to_L2_traffic"] - global_load_from_req

    Reg_to_global_reqs = results["Normal_global_st"] + results["Atom_global_st"] + results["Red_global_st"]
    Reg_to_shmem_reqs = results["Normal_shmem_st"] + results["Atom_shmem_st"]
    duration = results["Duration"] * 1e-9
    cycles = results["Cycles"]

    L1_util = results["L1_util"] * 0.01
    L1_bandwidth = L1_to_Reg_traffic / duration / L1_util
    L2_bandwidth = results["L2_throughput"] / (results["L2_util"] * 0.01)
    DRAM_bandwidth = results["DRAM_throughput"] / (results["DRAM_util"] * 0.01)

    # Write to CSV
    with open(output_csv, "w") as f:
        f.write("Item,Value\n")
        f.write(f"L1_to_Reg_traffic,{L1_to_Reg_traffic}\n")
        f.write(f"L1_to_Reg_traffic_calc,{L1_to_Reg_traffic_calc}\n")
        f.write(f"global_load_calc,{global_load_calc}\n")
        f.write(f"global_load_from_req,{global_load_from_req}\n")
        f.write(f"measured_traffic_delta,{measured_traffic_delta}\n")
        f.write(f"Reg_to_global_reqs,{Reg_to_global_reqs}\n")
        f.write(f"Reg_to_shmem_reqs,{Reg_to_shmem_reqs}\n")
        f.write(f"duration,{duration}\n")
        f.write(f"L1_util,{L1_util}\n")
        f.write(f"L1_bandwidth,{L1_bandwidth}\n")
        f.write(f"L2_bandwidth,{L2_bandwidth}\n")
        f.write(f"DRAM_bandwidth,{DRAM_bandwidth}\n")


if __name__ == "__main__":
    problem_shape = sys.argv[1]
    ncu_report_file = sys.argv[2]
    output_csv = sys.argv[3]
    extract_values(ncu_report_file, problem_shape, output_csv)