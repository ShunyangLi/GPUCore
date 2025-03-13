/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/10/10.
 */


#include "core.cuh"

__global__ auto scan_kernel(const int* d_degree, uint* buf_tails, uint* g_buffers,
                            int alpha, int beta, uint u_num, uint num_vertex) -> void {

    __shared__ uint* g_buffer;
    __shared__ uint bufTail;

    if (threadIdx.x == 0) {
        bufTail = 0;
        g_buffer = g_buffers + blockIdx.x * GLBUFFER_SIZE;
    }
    __syncthreads();

    uint g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint base = 0; base < num_vertex; base += N_THREADS) {
        uint v = base + g_idx;
        if (v >= num_vertex) continue;

        uint threshold = v < u_num ? alpha : beta;

        if (d_degree[v] < threshold) {
            uint idx = atomicAdd(&bufTail, 1);
            writeToBuffer(g_buffer, idx, v);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        buf_tails[blockIdx.x] = bufTail;
    }
}


__global__ auto peel_kernel(const uint* d_offset, const uint* d_neighbors, int* d_degree,
                            const uint* buf_tails, uint* g_buffers,
                            uint u_num, int alpha, int beta) -> void {

    __shared__ uint buf_tail;
    __shared__ uint* g_buffer;
    __shared__ uint base;

    uint warp_id = threadIdx.x / 32;
    uint lane_id = threadIdx.x % 32;
    uint regTail;
    uint i;

    if (threadIdx.x == 0) {
        buf_tail = buf_tails[blockIdx.x];
        base = 0;
        g_buffer = g_buffers + blockIdx.x * GLBUFFER_SIZE;
    }

    while (true) {
        __syncthreads();
        // all the threads will evaluate to true at same iteration
        if (base == buf_tail) break;
        i = base + warp_id;
        regTail = buf_tail;

        __syncthreads();

        if (i >= regTail) continue;// this warp won't have to do anything

        if (threadIdx.x == 0) {
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if (regTail < base) base = regTail;
        }

        //bufTail is incremented in the code below:
        uint v = readFromBuffer(g_buffer, i);

        uint start = d_offset[v];
        uint end = d_offset[v + 1];

        while (true) {
            __syncwarp();

            if (start >= end) break;

            uint j = start + lane_id;
            start += WARP_SIZE;
            if (j >= end) continue;

            uint u = d_neighbors[j];
            int threshold = u < u_num ? alpha : beta;

            int deg_u = atomicSub(d_degree + u, 1);

            if ((deg_u - 1) == (threshold - 1)) {
                uint loc = atomicAdd(&buf_tail, 1);
                writeToBuffer(g_buffer, loc, u);
            }
        }
    }
}

/**
 * abcore online peeling algorithm on gpu
 * @param g graph
 * @param alpha alpha value
 * @param beta beta value
 */
auto g_abcore_peeling(Graph* g, int alpha, int beta) -> double {

    log_info("running (alpha,beta)-core online peeling algorithm on GPU");

    auto left_degree_max = std::max_element(g->degrees, g->degrees + g->u_num);
    auto right_degree_max = std::max_element(g->degrees + g->u_num, g->degrees + g->n - 1);

    // check if the graph is valid
    if (*left_degree_max < alpha || *right_degree_max < beta) {
        log_error("max degree: (%d, %d), query (%d, %d) is not valid", *left_degree_max, *right_degree_max, alpha, beta);
        return 0;
    }

    uint* d_offset;
    uint* d_neighbors;
    int* d_degree;
    uint* g_buffers;
    uint* buf_tails;

    CER(cudaMalloc(&d_offset, sizeof(uint) * (g->n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g->m * 2));
    CER(cudaMalloc(&d_degree, sizeof(int) * g->n));
    CER(cudaMalloc(&buf_tails, sizeof(uint) * BLK_NUMS));
    CER(cudaMalloc(&g_buffers, sizeof(uint) * GLBUFFER_SIZE * BLK_NUMS));


    CER(cudaMemcpy((void*) d_offset, (void*) g->offsets, sizeof(uint) * (g->n + 1), cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_neighbors, (void*) g->neighbors, sizeof(uint) * g->m * 2, cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_degree, (void*) g->degrees, sizeof(uint) * g->n, cudaMemcpyHostToDevice));
    CER(cudaMemset((void*) buf_tails, 0, sizeof(uint) * BLK_NUMS));


    auto timer = new Timer();
    timer->reset();

    scan_kernel<<<BLK_NUMS, BLK_DIM>>>(d_degree, buf_tails, g_buffers, alpha, beta, g->u_num, g->n);

    cudaDeviceSynchronize();

    log_info("abcore scan time on gpu: %f s", timer->elapsed());

    timer->reset();
    peel_kernel<<<BLK_NUMS, BLK_DIM>>>(d_offset, d_neighbors, d_degree, buf_tails, g_buffers, g->u_num, alpha, beta);

    cudaDeviceSynchronize();

    auto time = timer->elapsed();
    log_info("abcore peeling time on gpu: %f s", time);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    auto upper_vertices = std::vector<uint>();
    auto lower_vertices = std::vector<uint>();
    auto degrees = new int[g->n];

    // get degrees
    cudaMemcpy((void*) degrees, (void*) d_degree, sizeof(int) * g->n, cudaMemcpyDeviceToHost);

    // get result
    for (auto i = 0; i < g->u_num; i++)
        if (degrees[i] >= alpha) upper_vertices.push_back(i);
    for (auto i = g->u_num; i < g->n; i++)
        if (degrees[i] >= beta) lower_vertices.push_back(i);

    // free cuda memory
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_degree);
    cudaFree(buf_tails);
    cudaFree(g_buffers);

    delete timer;

#ifdef DISPLAY_RESULT
    log_info("upper vertices: %d, lower vertices: %d", upper_vertices.size(), lower_vertices.size());
#endif

    return time;
}


__global__ auto scan_kernel_core(int* d_degree, uint* buf_tails, uint* g_buffers,
                            int alpha, int beta, uint u_num, uint num_vertex,
                                 const int* d_core, int k, bool* is_valid, const uint* d_offset, const uint* d_neighbors) -> void {

    __shared__ uint* g_buffer;
    __shared__ uint bufTail;

    if (threadIdx.x == 0) {
        bufTail = 0;
        g_buffer = g_buffers + blockIdx.x * GLBUFFER_SIZE;
    }
    __syncthreads();

    uint g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint base = 0; base < num_vertex; base += N_THREADS) {
        uint v = base + g_idx;
        if (v >= num_vertex) continue;

        if (d_core[v] < k) {
            is_valid[v] = false;
            uint threshold = v < u_num ? alpha : beta;

            if (d_degree[v] < threshold) {
                uint idx = atomicAdd(&bufTail, 1);
                writeToBuffer(g_buffer, idx, v);
            }
        } else if (d_core[v] >= k) {
            is_valid[v] = false;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        buf_tails[blockIdx.x] = bufTail;
    }
}


__global__ auto peel_kernel_core(const uint* d_offset, const uint* d_neighbors, int* d_degree,
                            const uint* buf_tails, uint* g_buffers,
                            uint u_num, int alpha, int beta, bool* is_valid) -> void {

    __shared__ uint buf_tail;
    __shared__ uint* g_buffer;
    __shared__ uint base;

    uint warp_id = threadIdx.x / 32;
    uint lane_id = threadIdx.x % 32;
    uint regTail;
    uint i;

    if (threadIdx.x == 0) {
        buf_tail = buf_tails[blockIdx.x];
        base = 0;
        g_buffer = g_buffers + blockIdx.x * GLBUFFER_SIZE;
    }

    while (true) {
        __syncthreads();
        // all the threads will evaluate to true at same iteration
        if (base == buf_tail) break;
        i = base + warp_id;
        regTail = buf_tail;

        __syncthreads();

        if (i >= regTail) continue;// this warp won't have to do anything

        if (threadIdx.x == 0) {
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if (regTail < base) base = regTail;
        }

        //bufTail is incremented in the code below:
        uint v = readFromBuffer(g_buffer, i);

        uint start = d_offset[v];
        uint end = d_offset[v + 1];

        while (true) {
            __syncwarp();

            if (start >= end) break;

            uint j = start + lane_id;
            start += WARP_SIZE;
            if (j >= end) continue;

            uint u = d_neighbors[j];
            if (!is_valid[u]) continue;

            int threshold = u < u_num ? alpha : beta;

            int deg_u = atomicSub(d_degree + u, 1);

            if ((deg_u - 1) == (threshold - 1)) {
                uint loc = atomicAdd(&buf_tail, 1);
                writeToBuffer(g_buffer, loc, u);
            }
        }
    }
}

/**
 * abcore online peeling algorithm on gpu
 * @param g graph
 * @param alpha alpha value
 * @param beta beta value
 */
auto g_abcore_peeling_core(Graph& g, int alpha, int beta) -> double {

    log_info("running (alpha,beta)-core online peeling algorithm on GPU based on kcore");

    auto left_degree_max = std::max_element(g.degrees, g.degrees + g.u_num);
    auto right_degree_max = std::max_element(g.degrees + g.u_num, g.degrees + g.n - 1);

    // check if the graph is valid
    if (*left_degree_max < alpha || *right_degree_max < beta) {
        log_error("max degree: (%d, %d), query (%d, %d) is not valid", *left_degree_max, *right_degree_max, alpha, beta);
        return 0;
    }

    bool* is_valid;
    uint* d_offset;
    uint* d_neighbors;
    int* d_degree;
    uint* g_buffers;
    uint* buf_tails;
    int* d_core;

    CER(cudaMalloc(&d_offset, sizeof(uint) * (g.n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g.m * 2));
    CER(cudaMalloc(&d_degree, sizeof(int) * g.n));
    CER(cudaMalloc(&buf_tails, sizeof(uint) * BLK_NUMS));
    CER(cudaMalloc(&g_buffers, sizeof(uint) * GLBUFFER_SIZE * BLK_NUMS));
    CER(cudaMalloc(&is_valid, sizeof(bool) * g.n));
    CER(cudaMalloc(&d_core, sizeof(int) * g.n));

    CER(cudaMemcpy((void*) d_offset, (void*) g.offsets, sizeof(uint) * (g.n + 1), cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_neighbors, (void*) g.neighbors, sizeof(uint) * g.m * 2, cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_degree, (void*) g.degrees, sizeof(uint) * g.n, cudaMemcpyHostToDevice));
    CER(cudaMemset((void*) buf_tails, 0, sizeof(uint) * BLK_NUMS));
    CER(cudaMemset((void*) is_valid, 1, sizeof(bool) * g.n));
    CER(cudaMemcpy((void*) d_core, (void*) g.core, sizeof(int) * g.n, cudaMemcpyHostToDevice));

    int k = std::min(alpha, beta);


    auto timer = new Timer();
    timer->reset();

    scan_kernel_core<<<BLK_NUMS, BLK_DIM>>>(d_degree, buf_tails, g_buffers, alpha, beta, g.u_num, g.n, d_core, k, is_valid, d_offset, d_neighbors);

    peel_kernel_core<<<BLK_NUMS, BLK_DIM>>>(d_offset, d_neighbors, d_degree, buf_tails, g_buffers, g.u_num, alpha, beta, is_valid);

    cudaDeviceSynchronize();

    auto time = timer->elapsed();
    log_info("abcore peeling time on gpu: %f s", time);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        log_trace("CUDA error: %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    auto upper_vertices = std::vector<uint>();
    auto lower_vertices = std::vector<uint>();
    auto degrees = new int[g.n];

    // get degrees
    cudaMemcpy((void*) degrees, (void*) d_degree, sizeof(int) * g.n, cudaMemcpyDeviceToHost);

    // get result
    for (auto i = 0; i < g.u_num; i++)
        if (degrees[i] >= alpha) upper_vertices.push_back(i);
    for (auto i = g.u_num; i < g.n; i++)
        if (degrees[i] >= beta) lower_vertices.push_back(i);

    // free cuda memory
    cudaFree(d_offset);
    cudaFree(d_neighbors);
    cudaFree(d_degree);
    cudaFree(buf_tails);
    cudaFree(g_buffers);

    delete timer;

#ifdef DISPLAY_RESULT
    log_info("upper vertices: %d, lower vertices: %d", upper_vertices.size(), lower_vertices.size());
#endif

    return time;
}

