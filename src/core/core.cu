/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/8/8.
 * @brief the main file of core decomposition
 */
#include "core.cuh"


__global__ auto peel_cores(const uint* d_offset, const uint* d_neighbors, int* d_degrees,
                           uint* d_currs, uint* d_nexts, uint* is_peels,
                           uint u_num, uint num_vertex, uint lower_max) -> void {
    // first step is to scan

    __shared__ uint* d_curr;
    __shared__ uint* d_next;
    __shared__ int* d_degree;
    __shared__ uint* is_peel;

    __shared__ uint d_curr_idx;
    __shared__ uint d_next_idx;
    __shared__ uint base;

    __shared__ int beta;

    // set alpha value
    int alpha = blockIdx.x + 1;

    uint warp_id = threadIdx.x / 32;
    uint lane_id = threadIdx.x % 32;
    uint regTail;
    uint i;

    if (threadIdx.x == 0) {
        d_curr = d_currs + blockIdx.x * num_vertex;
        d_next = d_nexts + blockIdx.x * num_vertex;
        d_degree = d_degrees + blockIdx.x * num_vertex;
        is_peel = is_peels + blockIdx.x * num_vertex;

        beta = 0;
    }

    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            d_curr_idx = 0;
            d_next_idx = 0;

            beta += 1;
            base = 0;
        }

        __syncthreads();

        if (beta >= 10) break;

        // then compute beta 1 - beta_max

        uint g_idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (uint u = 0; u < num_vertex; u += N_THREADS) {
            uint v = u + g_idx;
            if (v >= num_vertex) continue;

            uint threshold = v < u_num ? alpha : beta;

            if (d_degree[v] < threshold && is_peel[v] == 0) {
                uint idx = atomicAdd(&d_curr_idx, 1);
                d_currs[idx] = v;
            }
        }

        if (threadIdx.x == 0) {
            printf("alpha: %d, beta: %d, num: %d\n", alpha, beta, d_curr_idx);
        }

        __syncthreads();

        while (true) {
            __syncthreads();
            // all the threads will evaluate to true at same iteration
            if (base == d_curr_idx) break;
            i = base + warp_id;
            regTail = d_curr_idx;

            __syncthreads();

            if (i >= regTail) continue;// this warp won't have to do anything

            if (threadIdx.x == 0) {
                // update base for next iteration
                base += WARPS_EACH_BLK;
                if (regTail < base) base = regTail;
            }

            //bufTail is incremented in the code below:
            uint v = d_curr[i];

            uint start = d_offset[v];
            uint end = d_offset[v + 1];

            while (true) {
                __syncwarp();

                if (start >= end) break;

                uint j = start + lane_id;
                start += WARP_SIZE;
                if (j >= end) continue;

                uint u = d_neighbors[j];
                if (d_degree[u] > beta) {

                    int threshold = u < u_num ? alpha : beta;

                    int deg_u = atomicSub(d_degree + u, 1);

                    if (deg_u == threshold + 1 && is_peel[u] == 0) {
                        uint loc = atomicAdd(&d_next_idx, 1);
                        d_next[loc] = u;

                        // set peeled
                        atomicExch(&is_peel[u],1);
                    }

                    if (deg_u <= threshold) {
                        atomicAdd(&d_degree[u], 1);
                    }
                }

            }

            __syncthreads();

            // swap d_next and d_curr
            if (threadIdx.x == 0) {
                d_curr_idx = d_next_idx;
                d_curr = d_next;
                d_next_idx = 0;
                base = 0;
            }

            __syncthreads();
        }

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
    uint blk_num = free_memory  * 0.9 / (g->n * sizeof(int) * 4);

    blk_num = g->u_max_degree < blk_num ? g->u_max_degree : blk_num;

    log_info("block number: %d", blk_num);

    // allaoce degree for each block
    int *degrees;
    uint *currs;
    uint *nexts;
    uint *is_peels;

    CER(cudaMalloc(&degrees, sizeof(int) * g->n * blk_num));
    CER(cudaMalloc(&currs, sizeof(int) * g->n * blk_num));
    CER(cudaMalloc(&nexts, sizeof(int) * g->n * blk_num));
    CER(cudaMalloc(&is_peels, sizeof(int) * g->n * blk_num));

    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);

    for (int i = 0; i < blk_num; i++) {
        cudaMemcpy(degrees + i * g->n, g->degrees, sizeof(int) * g->n, cudaMemcpyHostToDevice);
    }

    cudaMemset((void*) is_peels, 0, sizeof(uint) * g->n * blk_num);


    log_info("block number: %d", blk_num);

    auto timer = new CudaTimer();
    timer->reset();

    peel_cores<<<1, BLK_DIM>>>(d_offset, d_neighbors, degrees, currs, nexts, is_peels,
            g->u_num, g->n, g->l_max_degree);

    cudaDeviceSynchronize();

    auto time = timer->elapsed();
    log_info("time: %f", time);

}