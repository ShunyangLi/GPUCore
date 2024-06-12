/**
 * @author Shunyang Li
 * Contact: sli@cse.unsw.edu.au
 * @date on 2023/12/22.
 * @brief: (alpha, beta)-core decomposition on CPU
 * from "Efficient (α, β)-core Computation: an Index-based Approach"
 */
#include "core.cuh"

void crossUpdate_for_kcore(std::vector<std::vector<uint>>& bicore, int alpha, int k_x, vid_t v) {
    for (int beta = k_x; beta > 0; beta--) {
        bool stop = false;
#pragma omp critical
        {
            if (bicore[v][beta] < alpha) {
                bicore[v][beta] = alpha;
            } else {
                stop = true;
            }
        }

        if (stop) break;
    }
}


auto peel_upper_bicore(int alpha, Graph& g, int thread, std::vector<std::vector<uint>>& bicore_u, std::vector<std::vector<uint>>& bicore_v) -> void {
    int dd_;
    int pre_upper_k = alpha - 1;

    auto tmp_degree = new int[g.n];
    // copy the degree value to tmp_degree

    for (int beta = 1; beta <= g.l_max_degree + 1; beta++) {
        int pre_ = beta - 1;
        bool stop = true;
        auto invalid_vertices = std::vector<uint>();
        {
#pragma omp parallel num_threads(thread) default(none) shared(g, invalid_vertices, alpha, dd_, pre_upper_k, stop, pre_, beta)
            {
#pragma omp for schedule(dynamic)
                for (auto v = 0; v < g.n; v++) {
                    if (g.degrees[v] <= 0) continue;
                    int threshold = v < g.u_num ? alpha : beta;
#pragma omp critical
                    {
                        stop = false;
                    };

                    if (g.degrees[v] < threshold) {
#pragma omp critical
                        invalid_vertices.push_back(v);
                    }
                }
            }
        }
#pragma omp barrier

        if (stop) break;

        auto next_invalid_vertices = std::vector<uint>();
        while (true) {
            {
#pragma omp parallel num_threads(thread) default(none) shared(g, invalid_vertices, alpha, dd_, pre_upper_k, stop, pre_, next_invalid_vertices, bicore_u, bicore_v)
#pragma omp for schedule(dynamic)
                for (auto const& u: invalid_vertices) {
                    if (g.degrees[u] <= 0) continue;
                    auto u_nbr_len = g.offsets[u + 1] - g.offsets[u];
                    auto u_nbr = g.neighbors + g.offsets[u];

                    for (auto i = 0; i < u_nbr_len; i++) {
                        auto const v = u_nbr[i];
                        if (g.degrees[v] <= 0) continue;
                        int threshold = v < g.u_num ? alpha - 1 : pre_;

                        auto old_deg = 0;
#pragma omp atomic capture
                        {
                            old_deg = g.degrees[v];
                            g.degrees[v]--;
                        }

                        if (old_deg - 1 == threshold) {
#pragma omp critical
                            next_invalid_vertices.push_back(v);
                            if (pre_ == 0) continue ;

                             // update the index for v
                            if (v >= g.u_num) {
                                crossUpdate_for_kcore(bicore_v, alpha, pre_, v - g.u_num);
                            } else {
#pragma omp critical
                                {
                                    if (bicore_u[v][alpha] < pre_)
                                        bicore_u[v][alpha] = pre_;
                                }
                            }
                        }
                    }
                    g.degrees[u] = 0;

                    if (pre_ != 0) {
                        if (u < g.u_num) continue ;
                        crossUpdate_for_kcore(bicore_v, alpha, pre_, u - g.u_num);
                    }
                }
            }
#pragma omp barrier
            if (next_invalid_vertices.empty()) break;
            std::swap(invalid_vertices, next_invalid_vertices);
            next_invalid_vertices.clear();
        }

        if (beta == 1) {
            std::copy(g.degrees, g.degrees + g.n, tmp_degree);
        }
        invalid_vertices.clear();
    }

    std::copy(tmp_degree, tmp_degree + g.n, g.degrees);
    g.u_max_degree = *std::max_element(g.degrees, g.degrees + g.u_num);
    g.l_max_degree = *std::max_element(g.degrees + g.u_num, g.degrees + g.n);
}


auto peel_lower_bicore(int beta, Graph& g, int thread, std::vector<std::vector<uint>>& bicore_u, std::vector<std::vector<uint>>& bicore_v) -> void {
    int dd_;
    int pre_upper_k = beta - 1;

    auto tmp_degree = new int[g.n];
    // copy the degree value to tmp_degree

    for (int alpha = 1; alpha <= g.l_max_degree + 1; alpha++) {
        int pre_ = alpha - 1;
        bool stop = true;
        auto invalid_vertices = std::vector<uint>();
        // scan upper
        {
#pragma omp parallel num_threads(thread) default(none) shared(g, invalid_vertices, dd_, pre_upper_k, stop, pre_, alpha, beta)
            {
#pragma omp for schedule(dynamic)
                for (auto u = 0; u < g.n; u++) {
                    if (g.degrees[u] <= 0) continue;
                    int threshold = u < g.u_num ? alpha : beta;
#pragma omp critical
                    {
                        stop = false;
                    };

                    if (g.degrees[u] < threshold) {
#pragma omp critical
                        invalid_vertices.push_back(u);
                    }
                }
            }
        }
#pragma omp barrier


        if (stop) break;

        auto next_invalid_vertices = std::vector<uint>();
        while (true) {
            {
#pragma omp parallel num_threads(thread) default(none) shared(g, invalid_vertices, beta, dd_, pre_upper_k, stop, pre_, next_invalid_vertices, bicore_u, bicore_v)
#pragma omp for schedule(dynamic)
                for (auto const& u: invalid_vertices) {
                    if (g.degrees[u] <= 0) continue;

                    auto u_nbr_len = g.offsets[u + 1] - g.offsets[u];
                    auto u_nbr = g.neighbors + g.offsets[u];

                    for (auto i = 0; i < u_nbr_len; i++) {
                        auto const v = u_nbr[i];
                        if (g.degrees[v] <= 0) continue;
                        int threshold = v < g.u_num ? beta - 1 : pre_;

                        auto old_deg = 0;
#pragma omp atomic capture
                        {
                            old_deg = g.degrees[v];
                            g.degrees[v]--;
                        }

                        if (old_deg - 1 == threshold) {
#pragma omp critical
                            next_invalid_vertices.push_back(v);
                            if (pre_ == 0) continue ;

                            if (v < g.u_num) {
                                crossUpdate_for_kcore(bicore_u, beta, pre_, v);
                            } else {
#pragma omp critical
                                {
                                    if (bicore_v[v - g.u_num][beta] < pre_)
                                        bicore_v[v - g.u_num][beta] = pre_;
                                }
                            }
                        }
                    }
#pragma omp critical
                    g.degrees[u] = 0;

                    if (pre_ != 0) {
                        if (u >= g.u_num) continue ;
                        crossUpdate_for_kcore(bicore_u, beta, pre_, u);
                    }
                }
            }
#pragma omp barrier
            if (next_invalid_vertices.empty()) break;
            std::swap(invalid_vertices, next_invalid_vertices);
            next_invalid_vertices.clear();
        }

        if (beta == 1) {
            std::copy(g.degrees, g.degrees + g.n, tmp_degree);
        }
        invalid_vertices.clear();
    }

    std::copy(tmp_degree, tmp_degree + g.n, g.degrees);
    g.u_max_degree = *std::max_element(g.degrees, g.degrees + g.u_num);
    g.l_max_degree = *std::max_element(g.degrees + g.u_num, g.degrees + g.n);
}

/**
 * @brief: (alpha, beta)-core decomposition on CPU
 * @param g
 */
auto c_abcore_decomposition(Graph* g, int thread) -> void {
    log_info("running (alpha, beta)-core decomposition on CPU with %d threads", thread);

    auto bicore_index_upper = std::vector<std::vector<uint>>(g->u_num);
    auto bicore_index_lower = std::vector<std::vector<uint>>(g->l_num);

    auto tmp_degree = new int[g->n];

    auto invalid_upper = std::vector<bool>(g->u_num, false);
    auto invalid_lower = std::vector<bool>(g->l_num, false);

    // initialize the bicore index
    for (auto u = 0; u < g->u_num; u++) {
        bicore_index_upper[u].resize(g->degrees[u] + 1, 0);
    }

    for (auto v = g->u_num; v < g->n; v++) {
        auto const v_offset = v - g->u_num;
        bicore_index_lower[v_offset].resize(g->degrees[v] + 1, 0);
    }

    auto upper_deg_max = g->u_max_degree;
    auto lower_deg_max = g->l_max_degree;
    std::copy(g->degrees, g->degrees + g->n, tmp_degree);

    // beta_s is the delta value
    auto timer = new Timer();
    timer->reset();
    auto beta_s = 0;
    for (auto alpha = 1; alpha <= g->u_max_degree; alpha++) {
        peel_upper_bicore(alpha, *g, thread, bicore_index_upper, bicore_index_lower);
        beta_s = 0;
        for (auto u = 0; u < g->u_num; u++) {
            if (g->degrees[u] <= alpha) continue ;
            beta_s = std::max(beta_s, int(bicore_index_upper[u][alpha]));
        }
        if (beta_s <= alpha) break;
//        std::cout << "alpha: " << alpha << " beta_s: " << beta_s << std::endl;
    }

    g->u_max_degree = upper_deg_max;
    g->l_max_degree = lower_deg_max;
    std::copy(tmp_degree, tmp_degree + g->n, g->degrees);

    // peel lower

    std::cout << "beta_s: " << beta_s << std::endl;

    for (auto beta = 1; beta <= beta_s; beta++) {
        peel_lower_bicore(beta, *g, thread, bicore_index_upper, bicore_index_lower);
    }
    auto time = timer->elapsed();
    log_info("abcore decomposition time on cpu: %f s", time);

//    for (auto u = 0; u < g->u_num; u ++) {
//        for (auto alpha = 1; alpha < bicore_index_upper[u].size(); alpha ++) {
//            if (bicore_index_upper[u][alpha] == 0) continue ;
//            std::cout << "u: " << u << " alpha: " << alpha << " beta: " << bicore_index_upper[u][alpha] << std::endl;
//        }
//    }
}
