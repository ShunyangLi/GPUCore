/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/8/8.
 * @brief the main file of core decomposition
 */
#include "core.cuh"


__global__ auto peel_cores(const uint* d_offset, const uint* d_neighbors, int* d_degree,
                           uint* d_currs, uint* d_nexts, uint* is_peels,
                           int alpha, uint u_num, uint num_vertex, uint lower_max) {
    // first step is to scan

    __shared__ uint* d_curr;
    __shared__ uint* d_next;
    __shared__ uint* is_peel;

    __shared__ uint d_curr_idx;
    __shared__ uint d_next_idx;

    __shared__ uint beta;

    uint warp_id = threadIdx.x / 32;
    uint lane_id = threadIdx.x % 32;
    uint regTail;
    uint i;

    if (threadIdx.x == 0) {
        d_curr = d_currs + blockIdx.x * num_vertex;
        d_next = d_nexts + blockIdx.x * num_vertex;
        is_peel = is_peels + blockIdx.x * num_vertex;

        d_curr_idx = 0;
        d_next_idx = 0;
        beta = 1;
    }

    __syncthreads();

    uint g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint base = 0; base < num_vertex; base += N_THREADS) {
        uint v = base + g_idx;
        if (v >= num_vertex) continue;

        uint threshold = v < u_num ? alpha : beta;

        if (d_degree[v] < threshold) {
            uint idx = atomicAdd(&d_curr_idx, 1);
            d_currs[idx] = v;
        }
    }

    __syncthreads();

    while (beta <= lower_max + 1) {

        if (threadIdx.x == 0) {
            beta += 1;
        }
        __syncthreads();
    }


}


/**
* for 1 .. alpha -> block
 *  for 1 .. beta -> warp
 * needs O(3n * block) memory
*/
auto core_decomposition(Graph* g) -> void {

    uint* d_offset;
    uint* d_neighbors;

    CER(cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2));

    size_t free_memory;
    cudaMemGetInfo(&free_memory, nullptr);
    uint blk_num = free_memory  * 0.96 / (g->n * 4);

    // allaoce degree for each block
    uint *degrees;
    uint *currs;
    uint *nexts;
    uint *is_peels;

    CER(cudaMalloc(&degrees, sizeof(uint) * g->n * blk_num));
    CER(cudaMalloc(&currs, sizeof(int) * g->n * blk_num));
    CER(cudaMalloc(&nexts, sizeof(int) * g->n * blk_num));
    CER(cudaMalloc(&is_peels, sizeof(int) * g->n * blk_num));

    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);

    for (int i = 0; i < blk_num; i++) {
        cudaMemcpy(degrees + i * g->n, g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice);
    }

    cudaMemset((void*) is_peels, 0, sizeof(uint) * g->n * blk_num);







    return;
}