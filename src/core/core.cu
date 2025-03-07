///**
// * @author Shunyang Li
// * Contact: sli@cse.unsw.edu.au
// * @date on 2024/8/8.
// * @brief the main file of core decomposition
// */
//#include "core.cuh"
//
//
//__global__ auto peel_cores(const uint* d_offset, const uint* d_neighbors, int* d_degrees,
//                           uint* d_currs, uint* d_nexts, uint* is_peels,
//                           uint u_num, uint num_vertex, uint lower_max) -> void {
//    // first step is to scan
//
//    __shared__ uint* d_curr;
//    __shared__ uint* d_next;
//    __shared__ int* d_degree;
//    __shared__ uint* is_peel;
//
//    __shared__ uint d_curr_idx;
//    __shared__ uint d_next_idx;
//
//    __shared__ int beta;
//
//    // set alpha value
//    int alpha = blockIdx.x + 1;
//
//    uint warp_id = threadIdx.x >> 5;
//    uint lane_id = threadIdx.x & 31;
//
//    if (threadIdx.x == 0) {
//        d_curr = d_currs + blockIdx.x * num_vertex;
//        d_next = d_nexts + blockIdx.x * num_vertex;
//        d_degree = d_degrees + blockIdx.x * num_vertex;
//        is_peel = is_peels + blockIdx.x * num_vertex;
//        beta = 0;
//    }
//
//    __syncthreads();
//
//    while (true) {
//
//        if (threadIdx.x == 0) {
//            d_curr_idx = 0;
//            d_next_idx = 0;
//            beta += 1;
//        }
//        __syncthreads();
//
//        if (beta >= lower_max + 1) break;
//
//        // then compute beta 1 - beta_max
//
//        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
//        uint stride = blockDim.x * gridDim.x;
//
//        for (uint u = u_num + idx; u < num_vertex; u += stride) {
//            if (u >= num_vertex) break;
//
//            if (d_degree[u] == beta) {
//                uint loc = atomicAdd(&d_curr_idx, 1);
//                d_currs[loc] = u;
//            }
//        }
//
//
//
//        __syncthreads();
//
//
//
//        while (true) {
//
//            __syncthreads();
//
//            if (d_curr_idx == 0) break;
//
//            // each warp process a vertex in d_curr
//
//            for (int i = warp_id; i < d_curr_idx; i += WARP_SIZE) {
//
//                int v = d_curr[i];
//
//                uint start = d_offset[v];
//                uint end = d_offset[v + 1];
//
//                for (int j = lane_id + start; j < end; j += WARP_SIZE) {
//
//                    uint u = d_neighbors[j];
//                    int threshold = u < u_num ? alpha : beta;
//
//                    if (d_degree[u] > threshold) {
//
//                        int deg_u = atomicSub(d_degree + u, 1);
//
//                        if (deg_u == threshold + 1) {
//                            uint loc = atomicAdd(&d_next_idx, 1);
//                            d_next[loc] = u;
//                        }
//
//                        if (deg_u <= threshold) {
//                            atomicAdd(&d_degree[u], 1);
//                        }
//                    }
//                }
//
//                __syncwarp();
//            }
//
//            __syncthreads();
//
//            // swap d_next and d_curr
//            if (threadIdx.x == 0) {
//                d_curr_idx = 0;
//                d_curr_idx = d_next_idx;
//                *d_curr = *d_next;
//                d_next_idx = 0;
//            }
//        }
//
//    }
//
//
//}
//
//
///**
//* for 1 .. alpha -> block
// *  for 1 .. beta -> warp
// * needs O(3n * block) memory
//*/
//auto core_decomposition(Graph* g) -> void {
//
//    uint* d_offset;
//    uint* d_neighbors;
//
//    CER(cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1)));
//    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2));
//
//    size_t free_memory;
//    cudaMemGetInfo(&free_memory, nullptr);
//    uint blk_num = free_memory  * 0.9 / (g->n * sizeof(int) * 4);
//
//    blk_num = g->u_max_degree < blk_num ? g->u_max_degree : blk_num;
//
//    log_info("block number: %d", blk_num);
//
//    // allaoce degree for each block
//    int *degrees;
//    uint *currs;
//    uint *nexts;
//    uint *is_peels;
//
//    CER(cudaMalloc(&degrees, sizeof(int) * g->n * blk_num));
//    CER(cudaMalloc(&currs, sizeof(int) * g->n * blk_num));
//    CER(cudaMalloc(&nexts, sizeof(int) * g->n * blk_num));
//    CER(cudaMalloc(&is_peels, sizeof(int) * g->n * blk_num));
//
//    cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice);
//    cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice);
//
//    for (int i = 0; i < blk_num; i++) {
//        cudaMemcpy(degrees + i * g->n, g->degrees, sizeof(int) * g->n, cudaMemcpyHostToDevice);
//    }
//
//    cudaMemset((void*) is_peels, 0, sizeof(uint) * g->n * blk_num);
//
//
////    log_info("block number: %d", blk_num);
//
//    log_info("u_num: %d, n: %d, l_max_degree: %d", g->u_num, g->n, g->l_max_degree);
//    auto timer = new CudaTimer();
//    timer->reset();
//
//    peel_cores<<<10, BLK_DIM>>>(d_offset, d_neighbors, degrees, currs, nexts, is_peels,
//            g->u_num, g->n, g->l_max_degree);
//
//    CER(cudaDeviceSynchronize());
//
//    // copy degree back
//    int* h_degrees = new int[g->n * blk_num];
//    cudaMemcpy((void*) h_degrees, (void*) degrees, sizeof(int) * g->n * blk_num, cudaMemcpyDeviceToHost);
//
//
//    auto time = timer->elapsed();
//    log_info("time: %f", time);
//
//}