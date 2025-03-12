/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/6/30.
 */

#pragma once
#ifndef GPU_PLEX_UTILITY_H
#define GPU_PLEX_UTILITY_H


#include <algorithm>
#include <argparse/argparse.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <locale>

#include "config.h"
#include "log.h"
#include "table.h"
#include "timer.cuh"
#include "uf.h"

//#include "dbg.h"

typedef unsigned int vid_t;
typedef int num_t;
typedef unsigned long long int ull;


typedef struct G_pointers {
    unsigned int* neighbors;
    unsigned int* neighbors_offset;
    unsigned int* degrees;
    unsigned int V;
} G_pointers;//graph related


static void HandleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n",
               cudaGetErrorString(err),
               file, line);
        exit(EXIT_FAILURE);
    }
}

#define CER(err) \
    (HandleError(err, __FILE__, __LINE__))

static auto add_args(argparse::ArgumentParser &parser) -> void {
    parser.add_argument("--device")
            .help("GPU Device ID (must be a positive integer)")
            .default_value(0)
            .action([](const std::string &value) { return std::stoi(value); });

    parser.add_argument("--device_info")
            .help("Display GPU device properties")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--graph")
            .help("Graph file path")
            .default_value("/")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--bin")
            .help("Output binary file path")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--cpu")
            .help("Run CPU algorithms")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--gpu")
            .help("Run GPU algorithms")
            .default_value(false)
            .implicit_value(true);

    parser.add_argument("--algo")
            .help("Algorithm to run")
            .default_value("msp")
            .action([](const std::string &value) { return value; });

    parser.add_argument("--threads")
            .help("Number of threads (must be a positive integer)")
            .default_value(1)
            .action([](const std::string &value) { return std::stoi(value); });

    parser.add_argument("--query")
            .help("Query file path")
            .default_value("/")
            .action([](const std::string &value) { return value; });
}


static auto get_device_info(int const device_id) -> void {
    cudaDeviceProp prop{};
    CER(cudaGetDeviceProperties(&prop, device_id));
    std::ostringstream oss;
    oss << std::left << std::setw(40) << "Property" << "Info\n";
    oss << std::string(50, '-') << '\n';

    oss << std::setw(40) << "Device Number" << device_id << '\n';
    oss << std::setw(40) << "Device name" << prop.name << '\n';
    oss << std::setw(40) << "Memory Bus Width (bits)" << prop.memoryBusWidth << '\n';
    oss << std::setw(40) << "Peak Memory Bandwidth (GB/s)"
        << 2.0 * prop.memoryClockRate * (float(prop.memoryBusWidth) / 8) / 1.0e6 << '\n';
    oss << std::setw(40) << "Total global memory (bytes)" << prop.totalGlobalMem << '\n';
    oss << std::setw(40) << "Total global memory (GB)"
        << float(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << '\n';
    oss << std::setw(40) << "Number of SMs" << prop.multiProcessorCount << '\n';
    oss << std::setw(40) << "Compute Capability" << prop.major << '.' << prop.minor << '\n';
    oss << std::setw(40) << "Shared memory per block (bytes)" << prop.sharedMemPerBlock << '\n';
    oss << std::setw(40) << "Max threads per SM" << prop.maxThreadsPerMultiProcessor << '\n';
    oss << std::setw(40) << "Total max threads"
        << prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount << '\n';

    std::cout << oss.str();
}


#endif//GPU_PLEX_UTILITY_H
