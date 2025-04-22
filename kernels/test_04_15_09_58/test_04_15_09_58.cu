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

template<int M, int D>
struct attend_params {
    static_assert(M == 16 || M == 32, "TILE_SIZE_M must be either 16 or 32");
    static_assert(D == 64 || D == 128, "TILE_SIZE_D must be either 64 or 128");
    
    template<typename T=bf16, typename L=row_l> using kv_tile = rt<T, TILE_SIZE_N, D, L>;
    template<typename T=bf16, typename L=row_l> using qo_tile = rt<T, M, D, L>;
    template<typename T=float> using attn_tile = rt<T, M, TILE_SIZE_N>;
    using shared_kv_tile = st_bf<TILE_SIZE_N, D>;
    using shared_qo_tile = st_bf<M, D>;
    using global_qkvo_layout = gl<bf16, -1, -1, -1, D>; // batch, depth, row, col
};

template<int M, int D>
struct globals {
    typename attend_params<M, D>::global_qkvo_layout Qg, Kg, Vg, Og;
};

template<int M, int D>
__launch_bounds__(WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals<M, D> g) {
    using params = attend_params<M, D>;
    
    const int batch = blockIdx.y, head = blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    typename params::shared_kv_tile (&k_smem)[PIPE_STAGES] = al.allocate<typename params::shared_kv_tile, PIPE_STAGES>();
    typename params::shared_kv_tile (&v_smem)[PIPE_STAGES] = al.allocate<typename params::shared_kv_tile, PIPE_STAGES>();
    typename params::shared_qo_tile (&qo_smem)[1] = al.allocate<typename params::shared_qo_tile, 1>();

    typename params::template kv_tile<bf16> k_reg;
    typename params::template qo_tile<bf16> q_reg;
    typename params::template kv_tile<bf16, col_l> v_reg;
    typename params::template qo_tile<float> o_reg;
    typename params::template attn_tile<float> att_block;
    typename params::template attn_tile<bf16> att_block_mma;
    typename params::template attn_tile<float>::col_vec max_vec_last, max_vec, norm_vec;
    
    // going through shared memory improves coalescing of dram reads.
    load<1, false>(qo_smem[0], g.Qg, {batch, 0, head, 0});
    __syncwarp();
    load(q_reg, qo_smem[0]);
    __syncthreads();
    if constexpr(D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);
    else if constexpr(D == 64) q_reg *= __float2bfloat16(0.125f * 1.44269504089f);

    max_vec = base_types::constants<float>::neg_infty();
    norm_vec = 0.f;
    o_reg = 0.f;
    // launch the load of the first k, v tiles
    int kv_blocks = g.Kg.depth() / TILE_SIZE_N, tic = 0;
    load_async<1, false>(k_smem[0], g.Kg, {batch, 0, head, 0});
    load_async<1, false>(v_smem[0], g.Vg, {batch, 0, head, 0});
    for (auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++, tic=(tic + 1) % PIPE_STAGES) {
        int next_load_idx = kv_idx + 1;
        if (next_load_idx * TILE_SIZE_N < g.Kg.depth()) {  // Remove the redundant multiplication with TILE_SIZE_N
            int next_tic = (tic + 1) % PIPE_STAGES;
            load_async<1, false>(k_smem[next_tic], g.Kg, {batch, next_load_idx, head, 0});
            load_async<1, false>(v_smem[next_tic], g.Vg, {batch, next_load_idx, head, 0});
            load_async_wait<1>();
        }
        else load_async_wait();
        __syncthreads();

        load(k_reg, k_smem[tic]);
        att_block = 0.f;
        mma<transpose::N, transpose::T>(att_block, q_reg, k_reg, att_block); // Q@K.T
        max_vec_last = max_vec;
        max_vec = max<axis::COL>(att_block, max_vec); 
        att_block = exp2(att_block - max_vec); 
        max_vec_last = exp2(max_vec_last - max_vec);
        norm_vec *= max_vec_last; 
        norm_vec = sum<axis::COL>(att_block, norm_vec); 
        att_block_mma = att_block; 
        load(v_reg, v_smem[tic]); 
        o_reg *= max_vec_last; 
        mma<transpose::N, transpose::N>(o_reg, att_block_mma, v_reg, o_reg);
    }

    o_reg /= norm_vec;
    __syncthreads();
    store(qo_smem[0], o_reg);
    __syncwarp();
    store<1, false>(g.Og, qo_smem[0], {batch, 0, head, 0});
}

template<int M, int D>
void run_attend_ker(globals<M, D> g) {
    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY) / 2;
    cudaFuncSetAttribute(
        attend_ker<M, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    cudaDeviceSynchronize();
    attend_ker<M, D><<<dim3(g.Qg.rows(), g.Qg.batch()), WARP_THREADS, mem_size>>>(g);
    cudaDeviceSynchronize();
    CudaCheckError();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

PYBIND11_MODULE(test_04_15_09_58, m) {
    m.doc() = "test_04_15_09_58 python module";
    
    // Expose the different template specializations
    py::bind_function<run_attend_ker<16, 64>>(m, "wrapped_attend_ker_16_64", 
        &globals<16, 64>::Qg, &globals<16, 64>::Kg, &globals<16, 64>::Vg, &globals<16, 64>::Og);
    
    py::bind_function<run_attend_ker<16, 128>>(m, "wrapped_attend_ker_16_128", 
        &globals<16, 128>::Qg, &globals<16, 128>::Kg, &globals<16, 128>::Vg, &globals<16, 128>::Og);
    
    py::bind_function<run_attend_ker<32, 64>>(m, "wrapped_attend_ker_32_64", 
        &globals<32, 64>::Qg, &globals<32, 64>::Kg, &globals<32, 64>::Vg, &globals<32, 64>::Og);
    
    py::bind_function<run_attend_ker<32, 128>>(m, "wrapped_attend_ker_32_128", 
        &globals<32, 128>::Qg, &globals<32, 128>::Kg, &globals<32, 128>::Vg, &globals<32, 128>::Og);
}