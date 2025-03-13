/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/12/16.
 */
#pragma once
#ifndef BITRUSS_CORE_CUH
#define BITRUSS_CORE_CUH

#include "graph/graph.h"



static __device__ inline auto writeToBuffer(uint* glBuffer, uint loc, uint v) -> void {
    assert(loc < GLBUFFER_SIZE);
    glBuffer[loc] = v;
}

static __device__ inline auto readFromBuffer(uint* glBuffer, uint loc) -> uint {
    assert(loc < GLBUFFER_SIZE);
    return glBuffer[loc];
}


auto c_abcore_peeling(Graph& g, int alpha, int beta) -> double;
auto g_abcore_peeling(Graph* g, int alpha, int beta) -> double;
auto c_abcore_decomposition(Graph* g, int thread) -> void;
auto abcore_decomposition(Graph* g) -> void;

auto kcore(Graph &g) -> void;


auto core_decomposition(Graph* g) -> void;
auto g_abcore_peeling_core(Graph& g, int alpha, int beta) -> double;
auto g_abcore_peeling_adv(Graph& g, int alpha, int beta) -> double;
auto g_abcore_peeling(Graph* g, int alpha, int beta) -> double;
//auto c_abcore_peeling(Graph &g, int alpha, int beta) -> double ;

#endif//BITRUSS_CORE_CUH
