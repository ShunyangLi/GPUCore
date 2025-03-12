
#include "core.cuh"

__global__ void selectNodesAtLevel1(int *degrees, unsigned int level, unsigned int V,
                                    unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int bufTail;
    int const THID = threadIdx.x;

    if(THID == 0) {
        bufTail = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
    __syncthreads();

    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned int base = 0; base < V; base += N_THREADS){

        unsigned int v = base + global_threadIdx;

        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(&bufTail, 1);
            writeToBuffer(glBuffer, loc, v);
        }
    }

    __syncthreads();

    if(THID == 0)
    {
        bufTails [blockIdx.x] = bufTail;
    }
}




__global__ void processNodes1(int *d_degree, int* d_offset, int* d_neighbors, int level, int V,
                              unsigned int* bufTails, unsigned int* glBuffers,
                              unsigned int *global_count){
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
    int const THID = threadIdx.x;

    unsigned int warp_id = THID / 32;
    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i;
    if(THID==0){
        bufTail = bufTails[blockIdx.x];
        base = 0;
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
        assert(glBuffer!=NULL);
    }

    while(true){
        __syncthreads(); //syncthreads must be executed by all the threads
        if(base == bufTail) break; // all the threads will evaluate to true at same iteration
        i = base + warp_id;
        regTail = bufTail;
        __syncthreads();

        if(i >= regTail) continue; // this warp won't have to do anything

        if(THID == 0){
            // base += min(WARPS_EACH_BLK, regTail-base)
            // update base for next iteration
            base += WARPS_EACH_BLK;
            if(regTail < base )
                base = regTail;
        }
        //bufTail is incremented in the code below:
        unsigned int v = readFromBuffer(glBuffer, i);
        unsigned int start = d_offset[v];
        unsigned int end = d_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_neighbors[j];
            if(*(d_degree +u) > level){

                unsigned int a = atomicSub(d_degree+u, 1);

                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);

                    writeToBuffer(glBuffer, loc, u);
                }

                if(a <= level){
                    // node degree became less than the level after decrementing...
                    atomicAdd(d_degree+u, 1);
                }
            }
        }

    }

    if(THID == 0 && bufTail>0){
        atomicAdd(global_count, bufTail); // atomic since contention among blocks
    }
}

auto kcore(Graph &g) -> void {

    // malloc graph memory on GPU like this
    int* d_offset;
    int* d_neighbors;
    int* d_degree;


    CER(cudaMalloc(&d_offset, sizeof(uint) * (g.n + 1)));
    CER(cudaMalloc(&d_neighbors, sizeof(uint) * g.m * 2));
    CER(cudaMalloc(&d_degree, sizeof(int) * g.n));

    CER(cudaMemcpy((void*) d_offset, (void*) g.offsets, sizeof(uint) * (g.n + 1), cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_neighbors, (void*) g.neighbors, sizeof(uint) * g.m * 2, cudaMemcpyHostToDevice));
    CER(cudaMemcpy((void*) d_degree, (void*) g.degrees, sizeof(uint) * g.n, cudaMemcpyHostToDevice));

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    CER(cudaMalloc(&global_count, sizeof(unsigned int)));
    CER(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    CER(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));

    auto timer = new Timer();
    while(count < g.n){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);

        selectNodesAtLevel1<<<BLK_NUMS, BLK_DIM>>>(d_degree, level,
                                                   g.n, bufTails, glBuffers);

        processNodes1<<<BLK_NUMS, BLK_DIM>>>(d_degree, d_offset, d_neighbors, level, g.n,
                                             bufTails, glBuffers, global_count);

        CER(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        // cout<<"*********Completed level: "<<level<<", global_count: "<<count<<" *********"<<endl;
        level++;
    }

    log_info("kcore time on gpu: %f s, core number: %d", timer->elapsed(), level-1);

    g.k_max = level-1;
    // copy back the degree array
    CER(cudaMemcpy((void*) g.core, (void*) d_degree, sizeof(uint) * g.n, cudaMemcpyDeviceToHost));

    // free the memory
    CER(cudaFree(d_offset));
    CER(cudaFree(d_neighbors));
    CER(cudaFree(d_degree));
    CER(cudaFree(global_count));
    CER(cudaFree(bufTails));
    CER(cudaFree(glBuffers));


}