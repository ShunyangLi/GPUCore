/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/12/16.
 * @brief: abcore online peeling algorithm on cpu
 */

#include "core.cuh"

#include <omp.h>

/**
 * abcore online peeling algorithm on cpu
 * @param g graph
 * @param alpha alpha value
 * @param beta beta value
 */
auto c_abcore_peeling(Graph &g, int alpha, int beta) -> double {

    log_info("running (alpha,beta)-core online peeling algorithm on CPU");

    auto left_degree_max = std::max_element(g.degrees, g.degrees + g.u_num);
    auto right_degree_max = std::max_element(g.degrees + g.u_num, g.degrees + g.n);

    // check if the graph is valid
    if (*left_degree_max < alpha || *right_degree_max < beta) {
        log_error("max degree: (%d, %d), query (%d, %d) is not valid", *left_degree_max, *right_degree_max, alpha,
                  beta);
        exit(EXIT_FAILURE);
    }

    auto upper_to_be_peeled = std::vector<uint>();
    auto lower_to_be_peeled = std::vector<uint>();

    auto invalid_upper = std::vector<bool>(g.u_num, false);
    auto invalid_lower = std::vector<bool>(g.l_num, false);

    // copy the degree value to upper_degrees and lower_degrees
    auto upper_degrees = std::vector<uint>(g.u_num);
    auto lower_degrees = std::vector<uint>(g.l_num);

    // copy the degree value to upper_degrees and lower_degrees
    std::copy(g.degrees, g.degrees + g.u_num, upper_degrees.begin());
    std::copy(g.degrees + g.u_num, g.degrees + g.n, lower_degrees.begin());

    auto timer = new Timer();
    timer->reset();

    // scan the graph and find the invalid nodes
    for (auto u = 0; u < g.u_num; u++) {
        if (g.degrees[u] < alpha) {
            upper_to_be_peeled.push_back(u);
        }
    }

    for (auto v = g.u_num; v < g.n; v++) {
        auto const v_offset = v - g.u_num;
        if (lower_degrees[v_offset] < beta) {
            lower_to_be_peeled.push_back(v_offset);
        }
    }

    // peel the graph and add the invalid vertex to the queue
    while (!upper_to_be_peeled.empty() || !lower_to_be_peeled.empty()) {

        // for upper vertices
        for (auto const &u: upper_to_be_peeled) {
            if (invalid_upper[u]) continue;

            auto const u_nbr_len = g.offsets[u + 1] - g.offsets[u];
            auto const u_nbr = g.neighbors + g.offsets[u];

            for (auto i = 0; i < u_nbr_len; i++) {
                auto v = u_nbr[i];
                auto const v_offset = v - g.u_num;
                if (invalid_lower[v_offset]) continue;
                lower_degrees[v_offset]--;
                if (lower_degrees[v_offset] == 0) invalid_lower[v_offset] = true;

                if (lower_degrees[v_offset] == beta - 1) {
                    lower_to_be_peeled.push_back(v_offset);
                }
            }

            upper_degrees[u] = 0;
            invalid_upper[u] = true;
        }

        upper_to_be_peeled.clear();

        // for lower vertices
        for (auto const &v_offset: lower_to_be_peeled) {
            auto v = v_offset + g.u_num;
            if (invalid_lower[v_offset]) continue;

            auto const v_nbr_len = g.offsets[v + 1] - g.offsets[v];
            auto const v_nbr = g.neighbors + g.offsets[v];

            for (auto i = 0; i < v_nbr_len; i++) {
                auto u = v_nbr[i];
                if (invalid_upper[u]) continue;
                upper_degrees[u]--;
                if (upper_degrees[u] == 0) invalid_upper[u] = true;

                if (upper_degrees[u] == alpha - 1) {
                    upper_to_be_peeled.push_back(u);
                }
            }
            lower_degrees[v_offset] = 0;
            invalid_lower[v_offset] = true;
        }
        lower_to_be_peeled.clear();
    }



    auto time = timer->elapsed();
    log_info("abcore peeling time on cpu: %f s", time);
//
//    auto upper_vertices = std::vector<uint>();
//    auto lower_vertices = std::vector<uint>();
//
//    for (auto u = 0; u < g.u_num; u++)
//        if (!invalid_upper[u]) upper_vertices.push_back(u);
//    for (auto v = g.u_num; v < g.n; v++)
//        if (!invalid_lower[v - g.u_num]) lower_vertices.push_back(v);
//
//    log_info("upper vertices: %d, lower vertices: %d", upper_vertices.size(), lower_vertices.size());
    return time;
}



/**
 * abcore online peeling algorithm on cpu
 * @param g graph
 * @param alpha alpha value
 * @param beta beta value
 */
auto c_abcore_peeling_mthreads(Graph &g, int alpha, int beta, int threads) -> double {

    log_info("running (alpha,beta)-core online peeling algorithm on CPU with %d threads", threads);

    omp_set_num_threads(threads);

    auto left_degree_max = std::max_element(g.degrees, g.degrees + g.u_num);
    auto right_degree_max = std::max_element(g.degrees + g.u_num, g.degrees + g.n);

    // check if the graph is valid
    if (*left_degree_max < alpha || *right_degree_max < beta) {
        log_error("max degree: (%d, %d), query (%d, %d) is not valid", *left_degree_max, *right_degree_max, alpha,
                  beta);
        exit(EXIT_FAILURE);
    }

    auto upper_to_be_peeled = std::vector<uint>();
    auto lower_to_be_peeled = std::vector<uint>();

    auto invalid_upper = std::vector<bool>(g.u_num, false);
    auto invalid_lower = std::vector<bool>(g.l_num, false);

    // copy the degree value to upper_degrees and lower_degrees
    auto upper_degrees = std::vector<int>(g.u_num);
    auto lower_degrees = std::vector<int>(g.l_num);

    // copy the degree value to upper_degrees and lower_degrees
    std::copy(g.degrees, g.degrees + g.u_num, upper_degrees.begin());
    std::copy(g.degrees + g.u_num, g.degrees + g.n, lower_degrees.begin());

    auto timer = new Timer();
    timer->reset();

    // scan the graph and find the invalid nodes
#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (auto u = 0; u < g.u_num; u++) {
            if (g.degrees[u] < alpha) {
#pragma omp critical
                upper_to_be_peeled.push_back(u);
            }
        }
    };


#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (auto v = g.u_num; v < g.n; v++) {
            auto const v_offset = v - g.u_num;
            if (lower_degrees[v_offset] < beta) {
#pragma omp critical
                lower_to_be_peeled.push_back(v_offset);
            }
        }
    };

    // peel the graph and add the invalid vertex to the queue
    while (!upper_to_be_peeled.empty() || !lower_to_be_peeled.empty()) {

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < upper_to_be_peeled.size(); i++) {
                auto u = upper_to_be_peeled[i];
                auto const u_nbr_len = g.offsets[u + 1] - g.offsets[u];
                auto const u_nbr = g.neighbors + g.offsets[u];

                for (auto i = 0; i < u_nbr_len; i++) {
                    auto v = u_nbr[i];
                    auto const v_offset = v - g.u_num;

                    // atomic operation
                    int new_val;
#pragma omp atomic capture
                    new_val = --lower_degrees[v_offset];

                    if (new_val == beta - 1) {
#pragma omp critical
                        lower_to_be_peeled.push_back(v_offset);
                    }
                }
                // atomic operation
#pragma omp atomic write
                upper_degrees[u] = 0;
            }
        };
        // for upper vertices
#pragma omp barrier // wait for all threads to finish the upper vertices

        upper_to_be_peeled.clear();

#pragma omp parallel
        {
#pragma omp for schedule(static)
            for (int i = 0; i < lower_to_be_peeled.size(); i++) {
                auto v_offset = lower_to_be_peeled[i];
                auto v = v_offset + g.u_num;

                auto const v_nbr_len = g.offsets[v + 1] - g.offsets[v];
                auto const v_nbr = g.neighbors + g.offsets[v];

                for (auto i = 0; i < v_nbr_len; i++) {
                    auto u = v_nbr[i];

                    int new_val;
#pragma omp atomic capture
                    new_val = --upper_degrees[u];

                    if (new_val == alpha - 1) {
#pragma omp critical
                        upper_to_be_peeled.push_back(u);
                    }
                }
#pragma omp atomic write
                lower_degrees[v_offset] = 0;
            }

        }
        // for lower vertices
#pragma omp barrier
        lower_to_be_peeled.clear();
    }



    auto time = timer->elapsed();
    log_info("abcore peeling time on cpu with %d threads: %f s", threads, time);
    //
    //    auto upper_vertices = std::vector<uint>();
    //    auto lower_vertices = std::vector<uint>();
    //
    //    for (auto u = 0; u < g.u_num; u++)
    //        if (!invalid_upper[u]) upper_vertices.push_back(u);
    //    for (auto v = g.u_num; v < g.n; v++)
    //        if (!invalid_lower[v - g.u_num]) lower_vertices.push_back(v);
    //
    //    log_info("upper vertices: %d, lower vertices: %d", upper_vertices.size(), lower_vertices.size());
    return time;
}
