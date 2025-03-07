/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2024/6/12.
 * @brief: main function
 */
#include <argparse/argparse.hpp>
#include <iostream>

#include "util/utility.h"
#include "graph/graph.h"
#include "core/core.cuh"

int main(int argc, char* argv[]) {

    argparse::ArgumentParser parser("core", "1.0.0");
    add_args(parser);

    std::locale loc("");
    std::locale::global(loc);

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        log_error("error: %s", err.what());
        std::cout << parser << std::endl;
        exit(EXIT_FAILURE);
    }

    auto device_count = 0;
    auto device_id = 0;

    cudaGetDeviceCount(&device_count);
    if (device_count == 0) log_warn("no gpu devices supporting CUDA.");

    if (parser.is_used("--device")) {
        device_id = parser.get<int>("--device");
        if (device_id >= device_count) {
            log_error("error: gpu device id %d is not available", device_id);
            exit(EXIT_FAILURE);
        }
        cudaSetDevice(device_id);
    }

    if (parser.get<bool>("--device_info")) {
        if (device_count == 0) log_warn("no gpu devices supporting CUDA.");
        else
            get_device_info(device_id);
    }

    if (parser.is_used("--graph")) {

        // convert the graph file to binary file
        if (parser.is_used("--bin")) {
            const std::string& filename = parser.get<std::string>("--bin");
            const std::string& dataset = parser.get<std::string>("--graph");

            auto g = Graph(dataset, true);
            g.graph_to_bin(filename);
            return 0;
        }


        auto dataset = parser.get<std::string>("--graph");
        auto g = Graph(dataset, false);

        auto pair = std::vector<std::pair<int, int>>({
                {1, 1},
                {2, 1},
                {2, 2},
                {3, 5},
                {10, 30},
                {55, 12},
                {4, 22},
                {20, 11}
        });

        for (auto& p : pair) {
            auto alpha = p.first;
            auto beta = p.second;
            g_abcore_peeling(&g, alpha, beta);
            c_abcore_peeling(g, alpha, beta);
        }

    }


    return 0;
}
