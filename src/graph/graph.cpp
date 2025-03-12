/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/10/3.
 */

#include "graph.h"

/**
 * load http://konect.cc/ graph data to binary file
 * @param graph_file graph file from konect
 * @param bin_file store the binary file
 * @param bin_file store the binary file
 */

auto Graph::process_graph(const std::string &path) -> void {
    log_info("processing graph file: %s", path.c_str());

    auto file = std::ifstream(path);
    std::string line;

    if (!file.is_open()) {
        log_error("Cannot open file %s\n", path.c_str());
        exit(EXIT_FAILURE);
    }

    auto upper_ids = std::vector<uint>(MAX_IDS, MAX_IDS);
    auto lower_ids = std::vector<uint>(MAX_IDS, MAX_IDS);

    auto upper_edges = std::vector<std::vector<uint>>();
    auto lower_edges = std::vector<std::vector<uint>>();


    while (std::getline(file, line)) {
        if (!line.empty() && line[0] == '%') continue;

        std::istringstream iss(line);
        int u, v;
        iss >> u >> v;

        if (upper_ids[u] == MAX_IDS) {
            upper_ids[u] = u_num++;
        }

        if (lower_ids[v] == MAX_IDS) {
            lower_ids[v] = l_num++;
        }

        upper_edges.resize(u_num);
        lower_edges.resize(l_num);

        upper_edges[upper_ids[u]].push_back(lower_ids[v]);
        lower_edges[lower_ids[v]].push_back(upper_ids[u]);

        m += 1;
    }

    file.close();

    // re-assign the vertices id for lower vertices
    auto offset = u_num;
    for (auto &nv: upper_edges) {
        for (auto &v: nv) {
            v += offset;
        }
    }

    // then merge the upper and lower vertices
    upper_edges.resize(u_num + l_num);

    for (auto i = 0; i < lower_edges.size(); i++) {
        upper_edges[i + offset] = lower_edges[i];
    }


    // sort the neighbor of each vertex with parallel
    {
#pragma omp parallel num_threads(THREADS)
        {
#pragma omp for schedule(dynamic)
            for (auto &upper_edge: upper_edges) {
                std::sort(upper_edge.begin(), upper_edge.end());
            }
        }
    }


    n = u_num + l_num;
    degrees = new uint[n];
    offsets = new uint[n + 1];
    neighbors = new uint[m * 2];

    // assign degrees
    for (auto u = 0; u < n; u++) degrees[u] = upper_edges[u].size();

    // assign max degree
    u_max_degree = *std::max_element(degrees, degrees + u_num);
    l_max_degree = *std::max_element(degrees + u_num, degrees + n);

    // assign offset
    offsets[0] = 0;
    for (auto i = 0; i < n; i++) offsets[i + 1] = offsets[i] + degrees[i];

    // assign neighbors
    auto all_neighbors = std::vector<uint>();
    for (auto u = 0; u < n; u++) {
        for (auto v: upper_edges[u]) {
            all_neighbors.push_back(v);
        }
    }

    assert(all_neighbors.size() == m * 2);
    std::copy(all_neighbors.begin(), all_neighbors.end(), neighbors);

    core = new int[n];
    k_max = 0;
    kcore(*this);


    {
#pragma omp parallel num_threads(THREADS)
        {
#pragma omp for schedule(dynamic)
            for (int u = 0; u < n; u ++) {
                // sort based on core value increasing order
                std::sort(neighbors + offsets[u], neighbors + offsets[u + 1], [this](uint a, uint b) {
                    return core[a] < core[b];
                });
            }
        }
    }


    log_info("graph with upper: %'d, lower: %'d, vertices: %'d, edges: %'d, core max: %d", u_num, l_num, n, m, k_max);
}

Graph::Graph(const std::string &filename, bool is_to_bin) {
    u_num = 0;
    l_num = 0;
    m = 0;
    n = 0;

    if (is_to_bin) {
        process_graph(filename);
    } else {
        if (filename.find(".bin") == std::string::npos) {
            log_error("graph file should be binary file");
            exit(EXIT_FAILURE);
        }
        load_graph_bin(filename);
    }
}

/**
 * convert graph to binary file
 * @param filename
 */
auto Graph::graph_to_bin(const std::string &filename) -> void {
    log_info("convert graph to binary file: %s", filename.c_str());

    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    out.write(reinterpret_cast<const char *>(&u_num), sizeof(u_num));
    out.write(reinterpret_cast<const char *>(&l_num), sizeof(l_num));
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(&u_max_degree), sizeof(u_max_degree));
    out.write(reinterpret_cast<const char *>(&l_max_degree), sizeof(l_max_degree));
    out.write(reinterpret_cast<const char *>(&m), sizeof(m));
    out.write(reinterpret_cast<const char *>(&k_max), sizeof(k_max));

    // write array member
    auto size = static_cast<std::streamsize>(sizeof(uint));
    out.write(reinterpret_cast<const char *>(neighbors), size * 2 * m);
    out.write(reinterpret_cast<const char *>(offsets), size * (u_num + l_num + 1));
    out.write(reinterpret_cast<const char *>(degrees), size * (u_num + l_num));
    out.write(reinterpret_cast<const char *>(core), size * n);


    out.close();
}

/**
 * load graph from binary file
 * @param filename
 */
auto Graph::load_graph_bin(const std::string &filename) -> void {
    log_info("loading graph from binary file: %s", filename.c_str());

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    // read single integer member
    in.read(reinterpret_cast<char *>(&u_num), sizeof(u_num));
    in.read(reinterpret_cast<char *>(&l_num), sizeof(l_num));
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    in.read(reinterpret_cast<char *>(&u_max_degree), sizeof(u_max_degree));
    in.read(reinterpret_cast<char *>(&l_max_degree), sizeof(l_max_degree));
    in.read(reinterpret_cast<char *>(&m), sizeof(m));
    in.read(reinterpret_cast<char *>(&k_max), sizeof(k_max));

    // allocate memory
    neighbors = new uint[2 * m];
    offsets = new uint[u_num + l_num + 1];
    degrees = new uint[u_num + l_num];
    core = new int[n];

    // read array member
    auto size = static_cast<std::streamsize>(sizeof(uint));

    in.read(reinterpret_cast<char *>(neighbors), size * 2 * m);
    in.read(reinterpret_cast<char *>(offsets), size * (u_num + l_num + 1));
    in.read(reinterpret_cast<char *>(degrees), size * (u_num + l_num));
    in.read(reinterpret_cast<char *>(core), size * n);

    in.close();

    core = new int[n];

    log_info("graph with upper: %'d, lower: %'d, vertices: %'d, edges: %'d, core max: %d", u_num, l_num, n, m, k_max);
}

Graph::~Graph() {
    delete[] neighbors;
    delete[] offsets;
    delete[] degrees;
}
