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
constexpr int TILE_SIZE_M = 32;
constexpr int TILE_SIZE_D = 128;
// constexpr int QO_SEQ = TILE_SIZE_N;
// constexpr int KV_BLOCKS = 32;
// constexpr int KV_SEQ = TILE_SIZE_N * KV_BLOCKS;

template<typename T=bf16, typename L=row_l> using kv_tile = rt<T, TILE_SIZE_N, TILE_SIZE_D, L>;
template<typename T=bf16, typename L=row_l> using qo_tile = rt<T, TILE_SIZE_M, TILE_SIZE_D, L>;
template<typename T=float> using attn_tile = rt<T, TILE_SIZE_M, TILE_SIZE_N>;
using shared_kv_tile = st_bf<TILE_SIZE_N, TILE_SIZE_D>;
using shared_qo_tile = st_bf<TILE_SIZE_M, TILE_SIZE_D>;
using global_qkvo_layout = gl<bf16, -1, -1, -1, TILE_SIZE_D>; // batch, depth, row, col
struct globals {
    global_qkvo_layout Qg, Kg, Vg, Og;
};

__launch_bounds__(WARP_THREADS, 1)
__global__ void attend_ker(const __grid_constant__ globals g) {
    const int batch = blockIdx.y, head = blockIdx.x;

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    shared_kv_tile (&k_smem)[PIPE_STAGES] = al.allocate<shared_kv_tile, PIPE_STAGES>();
    shared_kv_tile (&v_smem)[PIPE_STAGES] = al.allocate<shared_kv_tile, PIPE_STAGES>();
    shared_qo_tile (&qo_smem)[1] = al.allocate<shared_qo_tile, 1>();

    kv_tile<bf16> k_reg;
    qo_tile<bf16> q_reg;
    kv_tile<bf16, col_l> v_reg;
    qo_tile<float> o_reg;
    attn_tile<float> att_block;
    attn_tile<bf16> att_block_mma;
    typename attn_tile<float>::col_vec max_vec_last, max_vec, norm_vec;
    // going through shared memory improves coalescing of dram reads.
    load<1, false>(qo_smem[0], g.Qg, {batch, 0, head, 0});
    __syncwarp();
    load(q_reg, qo_smem[0]);
    __syncthreads();
    if constexpr(TILE_SIZE_D == 128) q_reg *= __float2bfloat16(0.08838834764f * 1.44269504089f);

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

void run_attend_ker(globals g) {
    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY) / 2;// PIPE_STAGES * TILE_SIZE_N * TILE_SIZE_D * 2 * 2 + TILE_SIZE_M * TILE_SIZE_D * 2 * 2; 
    cudaFuncSetAttribute(
        attend_ker,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );
    cudaDeviceSynchronize();
    attend_ker<<<dim3(g.Qg.rows(), g.Qg.batch()), WARP_THREADS, mem_size>>>(g);
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
    py::bind_function<run_attend_ker>(m, "wrapped_attend_ker", &globals::Qg, &globals::Kg, &globals::Vg, &globals::Og);
}