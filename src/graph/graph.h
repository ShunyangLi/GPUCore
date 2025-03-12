/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/10/3.
 */

#pragma once
#ifndef BITRUSS_GRAPH_H
#define BITRUSS_GRAPH_H

#include "util/utility.h"


class Graph {
public:
    explicit Graph(const std::string& filename, bool is_bin);
    auto graph_to_bin(const std::string& filename) -> void;
    auto load_graph_bin(const std::string& filename) -> void;
    ~Graph();

private:
    auto process_graph(const std::string& path) -> void;
public:
    // upper/lower vertices number, upper vertices max id, edge number
    uint u_num{}, l_num{}, n{}, m{}, k_max{};
    uint l_max_degree{}, u_max_degree{};
    uint* neighbors{};   // neighbors array, length = 2 * m
    uint* offsets{};     // offsets array, length = u_num + l_num + 1
    uint* degrees{};     // degrees array, length = u_num + l_num
    int* core{};         // core array, length = u_num + l_num
};

#include "core/core.cuh"

#endif//BITRUSS_GRAPH_H
