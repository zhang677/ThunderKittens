metrics = [
    "sass__inst_executed_shared_loads", # Number of shared memory load instructions executed other than LDSM
    "smsp__inst_executed_op_ldsm.sum", # LDSM: shmem to register
    "smsp_inst_executed_op_ldgsts.sum", # LDGSTS requests
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", # # of requests sent to T-Stage for global loads
    "sm__sass_l1tex_m_xbar2l1tex_read_bytes_mem_global_op_ldgsts_cache_bypass.sum", # LDGSTS traffic (L2 to shmem)
    "l1tex__m_xbar2l1tex_read_bytes.sum", # # of bytes read from L2 into L1TEX M-Stage
    "dram__bytes_read.sum", # # of bytes read from DRAM
    "l1tex__m_xbar2l1tex_read_bytes.sum.per_second", # L2 measured throughput
    "dram__bytes_read.per_second", # DRAM measured throughput 
]

shmem_load = metrics[0] * 16 + metrics[1] * 512