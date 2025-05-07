#include <iostream>
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

using namespace kittens;

constexpr int PIPE_STAGES = 2;
constexpr int TILE_SIZE_N = 16;
// TILE_SIZE_M and TILE_SIZE_D are now template parameters

template<int M, int D, int NUM_WORKERS>
struct attend_params {
    static_assert(M == 16 || M == 32 || M == 48 || M == 64, "TILE_SIZE_M must be either {16, 32, 48, 64}");
    static_assert(D == 64 || D == 96 || D == 128 || D == 160, "TILE_SIZE_D must be either {64, 96, 128, 160}");
    
    template<typename T=bf16, typename L=row_l> using kv_tile = rt<T, TILE_SIZE_N, D, L>;
    template<typename T=bf16, typename L=row_l> using qo_tile = rt<T, M, D, L>;
    template<typename T=float> using attn_tile = rt<T, M, TILE_SIZE_N>;
    using shared_kv_tile = st_bf<TILE_SIZE_N, D>;
    using shared_qo_tile = st_bf<M, D>;
    using global_qkvo_layout = gl<bf16, -1, -1, NUM_WORKERS, D>; // batch, depth, row, col
};

template<int M, int D, int NUM_WORKERS>
struct globals {
    typename attend_params<M, D, NUM_WORKERS>::global_qkvo_layout Qg, Kg, Vg, Og;
};

template<int M, int D, int NUM_WORKERS>
__launch_bounds__(NUM_WORKERS*WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<M, D, NUM_WORKERS> g) {
    using params = attend_params<M, D, NUM_WORKERS>;
    
    const int batch = blockIdx.x, head = kittens::warpid();

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    typename params::shared_kv_tile (&k_smem)[NUM_WORKERS][PIPE_STAGES] = al.allocate<typename params::shared_kv_tile, NUM_WORKERS, PIPE_STAGES>();
    typename params::shared_kv_tile (&v_smem)[NUM_WORKERS][PIPE_STAGES] = al.allocate<typename params::shared_kv_tile, NUM_WORKERS, PIPE_STAGES>();
    typename params::shared_qo_tile (&qo_smem)[NUM_WORKERS] = al.allocate<typename params::shared_qo_tile, NUM_WORKERS>();

    typename params::template kv_tile<bf16> k_reg;
    typename params::template qo_tile<bf16> q_reg;
    typename params::template kv_tile<bf16, col_l> v_reg;
    typename params::template qo_tile<float> o_reg;
    typename params::template attn_tile<float> att_block;
    typename params::template attn_tile<bf16> att_block_mma;
    typename params::template attn_tile<float>::col_vec max_vec_last, max_vec, norm_vec;
    
    // going through shared memory improves coalescing of dram reads.
    load<1, false>(qo_smem[head], g.Qg, {batch, 0, head, 0});
    __syncwarp();
    load(q_reg, qo_smem[head]);
    __syncthreads();
    if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);
    else if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);
    else if constexpr(D == 96) q_reg *= __float2bfloat16(0.10206207262f * 1.44269504089f);
    else if constexpr(D == 160) q_reg *= __float2bfloat16(0.07905694151f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = g.Kg.depth() / TILE_SIZE_N, tic = 0;
    load_async<1, false>(k_smem[head][0], g.Kg, {batch, 0, head, 0});
    load_async<1, false>(v_smem[head][0], g.Vg, {batch, 0, head, 0});
    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic + 1) % PIPE_STAGES) {
        int next_load_idx = kv_idx + 1;
        if (next_load_idx * TILE_SIZE_N < g.Kg.depth()) {  // Remove the redundant multiplication with TILE_SIZE_N
            int next_tic = (tic + 1) % PIPE_STAGES;
            load_async<1, false>(k_smem[head][next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_async<1, false>(v_smem[head][next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>();
        }
        else load_async_wait();
        __syncthreads();

        load(k_reg, k_smem[head][tic]);
        att_block = 0.f;
        mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
        max_vec_last = max_vec;
        max_vec = max<axis::COL>(att_block, max_vec); 
        att_block = exp2(att_block - max_vec); // M * Tn
        max_vec_last = exp2(max_vec_last - max_vec); // M
        norm_vec *= max_vec_last; 
        norm_vec = sum<axis::COL>(att_block, norm_vec);  // M * Tn
        att_block_mma = att_block; 
        load(v_reg, v_smem[tic]); 
        o_reg *= max_vec_last;  // M * d
        mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
    }

    o_reg /= norm_vec;
    __syncthreads();
    store(qo_smem[head], o_reg);
    __syncwarp();
    store<1, false>(g.Og, qo_smem[head], {batch, 0, head, 0});
}

template<int M, int D, int NUM_WORKERS>
void run_attend_ker(globals<M, D, NUM_WORKERS> g) {
    unsigned long mem_planned = 
    (2 * PIPE_STAGES * attend_params<M, D, NUM_WORKERS>::shared_kv_tile::num_elements * 
     sizeof(typename attend_params<M, D, NUM_WORKERS>::shared_kv_tile::dtype)) +
    (attend_params<M, D, NUM_WORKERS>::shared_qo_tile::num_elements * 
     sizeof(typename attend_params<M, D, NUM_WORKERS>::shared_qo_tile::dtype));
    
    unsigned long mem_size = mem_planned * NUM_WORKERS;
    cudaFuncSetAttribute(
        attend_ker<M, D, NUM_WORKERS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    cudaDeviceSynchronize();
    attend_ker<M, D, NUM_WORKERS><<<dim3(g.Qg.batch()), NUM_WORKERS * WARP_THREADS, mem_size>>>(g);
    cudaDeviceSynchronize();
    CudaCheckError();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

PYBIND11_MODULE(test_05_05_13_28, m) {
    m.doc() = "test_05_05_13_28 python module";
    
    // Expose the different template specializations
    py::bind_function<run_attend_ker<16, 64, 12>>(m, "wrapped_attend_ker_16_64_12", 
        &globals<16, 64, 12>::Qg, &globals<16, 64, 12>::Kg, &globals<16, 64, 12>::Vg, &globals<16, 64, 12>::Og);
    py::bind_function<run_attend_ker<16, 96, 8>>(m, "wrapped_attend_ker_16_96_8", 
        &globals<16, 96, 8>::Qg, &globals<16, 96, 8>::Kg, &globals<16, 96, 8>::Vg, &globals<16, 96, 8>::Og);  
    py::bind_function<run_attend_ker<16, 128, 7>>(m, "wrapped_attend_ker_16_128_7", 
        &globals<16, 128, 7>::Qg, &globals<16, 128, 7>::Kg, &globals<16, 128, 7>::Vg, &globals<16, 128, 7>::Og);
    py::bind_function<run_attend_ker<16, 160, 4>>(m, "wrapped_attend_ker_16_160_4", 
        &globals<16, 160, 4>::Qg, &globals<16, 160, 4>::Kg, &globals<16, 160, 4>::Vg, &globals<16, 160, 4>::Og);

    py::bind_function<run_attend_ker<32, 64, 8>>(m, "wrapped_attend_ker_32_64_8", 
        &globals<32, 64, 8>::Qg, &globals<32, 64, 8>::Kg, &globals<32, 64, 8>::Vg, &globals<32, 64, 8>::Og);
    py::bind_function<run_attend_ker<32, 96, 8>>(m, "wrapped_attend_ker_32_96_8", 
        &globals<32, 96, 8>::Qg, &globals<32, 96, 8>::Kg, &globals<32, 96, 8>::Vg, &globals<32, 96, 8>::Og);
    py::bind_function<run_attend_ker<32, 128, 6>>(m, "wrapped_attend_ker_32_128_6", 
        &globals<32, 128, 6>::Qg, &globals<32, 128, 6>::Kg, &globals<32, 128, 6>::Vg, &globals<32, 128, 6>::Og);
    py::bind_function<run_attend_ker<32, 160, 5>>(m, "wrapped_attend_ker_32_160_5", 
        &globals<32, 160, 5>::Qg, &globals<32, 160, 5>::Kg, &globals<32, 160, 5>::Vg, &globals<32, 160, 5>::Og);

    py::bind_function<run_attend_ker<48, 64, 8>>(m, "wrapped_attend_ker_48_64_8", 
        &globals<48, 64, 8>::Qg, &globals<48, 64, 8>::Kg, &globals<48, 64, 8>::Vg, &globals<48, 64, 8>::Og);
    py::bind_function<run_attend_ker<48, 96, 7>>(m, "wrapped_attend_ker_48_96_7", 
        &globals<48, 96, 7>::Qg, &globals<48, 96, 7>::Kg, &globals<48, 96, 7>::Vg, &globals<48, 96, 7>::Og);
    py::bind_function<run_attend_ker<48, 128, 5>>(m, "wrapped_attend_ker_48_128_5", 
        &globals<48, 128, 5>::Qg, &globals<48, 128, 5>::Kg, &globals<48, 128, 5>::Vg, &globals<48, 128, 5>::Og);
    py::bind_function<run_attend_ker<48, 160, 4>>(m, "wrapped_attend_ker_48_160_4", 
        &globals<48, 160, 4>::Qg, &globals<48, 160, 4>::Kg, &globals<48, 160, 4>::Vg, &globals<48, 160, 4>::Og);

    py::bind_function<run_attend_ker<64, 64, 8>>(m, "wrapped_attend_ker_64_64_8", 
        &globals<64, 64, 8>::Qg, &globals<64, 64, 8>::Kg, &globals<64, 64, 8>::Vg, &globals<64, 64, 8>::Og);
    py::bind_function<run_attend_ker<64, 96, 6>>(m, "wrapped_attend_ker_64_96_6", 
        &globals<64, 96, 6>::Qg, &globals<64, 96, 6>::Kg, &globals<64, 96, 6>::Vg, &globals<64, 96, 6>::Og);
    py::bind_function<run_attend_ker<64, 128, 4>>(m, "wrapped_attend_ker_64_128_4", 
        &globals<64, 128, 4>::Qg, &globals<64, 128, 4>::Kg, &globals<64, 128, 4>::Vg, &globals<64, 128, 4>::Og);
    py::bind_function<run_attend_ker<64, 160, 3>>(m, "wrapped_attend_ker_64_160_3", 
        &globals<64, 160, 3>::Qg, &globals<64, 160, 3>::Kg, &globals<64, 160, 3>::Vg, &globals<64, 160, 3>::Og);
}