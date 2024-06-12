/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/12/16.
 */
#pragma once
#ifndef BITRUSS_CORE_CUH
#define BITRUSS_CORE_CUH

#include "../graph/graph.h"

auto c_abcore_peeling(Graph& g, int alpha, int beta) -> void;
auto g_abcore_peeling(Graph* g, int alpha, int beta) -> void;
auto c_abcore_decomposition(Graph* g, int thread) -> void;
auto abcore_decomposition(Graph* g) -> void;

#endif//BITRUSS_CORE_CUH
