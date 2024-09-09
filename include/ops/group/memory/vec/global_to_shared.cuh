/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared vectors from and storing to global memory. 
 */

/**
 * @brief Loads data from global memory into shared memory vector.
 * 
 * This function loads data from a global memory location pointed to by `src` into a shared memory vector `dst`.
 * It calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 * 
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Reference to the shared vector where the data will be loaded.
 * @param src Pointer to the global memory location from where the data will be loaded.
 */
template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void load(SV &dst, const GL &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<SV>(idx);
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            float4 tmp;
            move<float4>::ldg(tmp, &src_ptr[i*elem_per_transfer]);
            move<float4>::sts(&dst[i*elem_per_transfer], tmp);
        }
    }
}

/**
 * @brief Stores data from a shared memory vector to global memory.
 * 
 * This function stores data from a shared memory vector `src` to a global memory location pointed to by `dst`.
 * Similar to the load function, it calculates the number of elements that can be transferred in one operation based on the size ratio of `float4` to the data type of `SV`.
 * The function ensures coalesced memory access and efficient use of bandwidth by dividing the work among threads in a warp.
 * 
 * @tparam SV Shared vector type, must satisfy ducks::sv::all concept.
 * @param dst Pointer to the global memory location where the data will be stored.
 * @param src Reference to the shared vector from where the data will be stored.
 */
template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void store(GL &dst, const SV &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = src.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst.template get<SV>(idx);
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < src.length) {
            float4 tmp;
            move<float4>::lds(tmp, &src[i*elem_per_transfer]);
            move<float4>::stg(&dst_ptr[i*elem_per_transfer], tmp);
        }
    }
}

template<ducks::sv::all SV, ducks::gl::all GL>
__device__ static inline void load_async(SV &dst, const GL &src, const index &idx) {
    constexpr int elem_per_transfer = sizeof(float4) / sizeof(typename SV::dtype);
    constexpr int total_calls = dst.length / elem_per_transfer; // guaranteed to divide
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<SV>(idx);
    #pragma unroll
    for(int i = threadIdx.x%GROUP_THREADS; i < total_calls; i+=GROUP_THREADS) {
        if(i * elem_per_transfer < dst.length) {
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                :
                : "l"((uint64_t)&dst[i*elem_per_transfer]), "l"((uint64_t)&src_ptr[i*elem_per_transfer])
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}