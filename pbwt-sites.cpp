#include <vector>
#include <numeric>  // For iota
#include <algorithm>  // For copy, min, max
#include <omp.h>  // For OpenMP
#include <memory>  // For unique_ptr
#include <cstring> // For memcpy

#include "util.h"

using namespace std;

vector<vector<int>> build_and_match(const vector<vector<int>>& hap_map_rows_haps, int num_threads, int L_param) {
    const int M = hap_map_rows_haps.size();
    if (M == 0) return {};
    const int N_sites = hap_map_rows_haps[0].size();
    if (N_sites == 0) return {};

    // Khởi tạo cấu trúc dữ liệu ban đầu
    vector<int> a(M);
    iota(a.begin(), a.end(), 0);
    vector<int> d(M, 0);

    int i0_alg3_state = 0;

    // Cấu hình OpenMP
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    const int actual_threads = min(num_threads, omp_get_max_threads());

    // Thread-local storage for matches để tránh xung đột khi ghi
    vector<vector<vector<int>>> thread_local_matches(actual_threads);
    for (auto& tlm : thread_local_matches) {
        tlm.reserve(min(10000, M * M / (4 * actual_threads)));
    }

    // Cache alleles để tối ưu truy cập bộ nhớ
    vector<vector<int>> cached_alleles;
    #pragma omp parallel
    {
        #pragma omp single
        {
            if (M * N_sites <= 100000000) { // Chỉ cache khi kích thước phù hợp
                cached_alleles.resize(M, vector<int>(N_sites));
                #pragma omp task
                {
                    for (int i = 0; i < M; i++) {
                        for (int j = 0; j < N_sites; j++) {
                            cached_alleles[i][j] = hap_map_rows_haps[i][j];
                        }
                    }
                }
            }
        }
    }

    // Dùng con trỏ để truy cập nhanh hơn
    const auto& alleles = cached_alleles.empty() ? hap_map_rows_haps : cached_alleles;

    for (int k_current_site = 0; k_current_site < N_sites; ++k_current_site) {
        // Algorithm 3: ReportLongMatches - đã tối ưu hóa
        {
            int u_alg3_block_count0 = 0;
            int v_alg3_block_count1 = 0;
            const int k_minus_L = k_current_site - L_param;

            for (int i_alg3_scan = 0; i_alg3_scan < M; ++i_alg3_scan) {
                const bool condition_block_break = (d[i_alg3_scan] > max(0, k_minus_L));

                if (condition_block_break) {
                    if (u_alg3_block_count0 && v_alg3_block_count1) {
                        const int block_size = i_alg3_scan - i0_alg3_state;

                        // Luôn song song hóa khi có ít nhất 1 phần tử
                        if (block_size >= 1) {
                            // Sử dụng grain size linh hoạt dựa trên block size
                            const int grain_size = max(1, min(16, block_size / (2 * actual_threads)));

                            #pragma omp parallel for schedule(dynamic, grain_size) num_threads(actual_threads)
                            for (int ia_loop = i0_alg3_state; ia_loop < i_alg3_scan; ++ia_loop) {
                                const int tid = omp_get_thread_num();
                                const int hap_ia = a[ia_loop];
                                const int allele_ia = alleles[hap_ia][k_current_site];

                                int dmin_loop_private = 0;
                                for (int ib_loop = ia_loop + 1; ib_loop < i_alg3_scan; ++ib_loop) {
                                    const int hap_ib = a[ib_loop];
                                    dmin_loop_private = max(dmin_loop_private, d[ib_loop]);

                                    if (alleles[hap_ib][k_current_site] != allele_ia) {
                                        const int len = k_current_site - dmin_loop_private;
                                        if (len >= L_param) {
                                            thread_local_matches[tid].push_back({
                                                k_current_site,
                                                min(hap_ia, hap_ib),
                                                max(hap_ia, hap_ib),
                                                len
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    u_alg3_block_count0 = 0;
                    v_alg3_block_count1 = 0;
                    i0_alg3_state = i_alg3_scan;
                }

                // Tránh cache miss với prefetch
                const int current_allele = alleles[a[i_alg3_scan]][k_current_site];
                if (current_allele == 0) {
                    u_alg3_block_count0++;
                } else {
                    v_alg3_block_count1++;
                }
            }

            // Process the last block
            if (u_alg3_block_count0 && v_alg3_block_count1) {
                const int block_size = M - i0_alg3_state;

                if (block_size >= 1) {
                    // Sử dụng grain size linh hoạt
                    const int grain_size = max(1, min(16, block_size / (2 * actual_threads)));

                    #pragma omp parallel for schedule(dynamic, grain_size) num_threads(actual_threads)
                    for (int ia_loop = i0_alg3_state; ia_loop < M; ++ia_loop) {
                        const int tid = omp_get_thread_num();
                        const int hap_ia = a[ia_loop];
                        const int allele_ia = alleles[hap_ia][k_current_site];

                        int dmin_loop_private = 0;
                        for (int ib_loop = ia_loop + 1; ib_loop < M; ++ib_loop) {
                            const int hap_ib = a[ib_loop];
                            dmin_loop_private = max(dmin_loop_private, d[ib_loop]);

                            if (alleles[hap_ib][k_current_site] != allele_ia) {
                                const int len = k_current_site - dmin_loop_private;
                                if (len >= L_param) {
                                    thread_local_matches[tid].push_back({
                                        k_current_site,
                                        min(hap_ia, hap_ib),
                                        max(hap_ia, hap_ib),
                                        len
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }

        // Algorithm 2: BuildPrefixAndDivergenceArrays
        // Tối ưu hóa song song xây dựng mảng prefix và divergence
        vector<int> next_a(M);
        vector<int> next_d(M);

        // Cách tiếp cận lock-free để xây dựng mảng kết quả
        if (M >= actual_threads * 2) {
            // Mảng để lưu vị trí bắt đầu cho mỗi thread
            vector<int> counts_0(actual_threads + 1, 0);
            vector<int> counts_1(actual_threads + 1, 0);

            // Bước 1: Tính số lượng phần tử mỗi loại trên từng thread
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                // Phân chia dữ liệu theo khối
                const int chunk_size = (M + actual_threads - 1) / actual_threads;
                const int start = tid * chunk_size;
                const int end = min(M, start + chunk_size);

                int count_0 = 0;
                int count_1 = 0;

                for (int i = start; i < end; ++i) {
                    const int current_hap = a[i];
                    const int allele = alleles[current_hap][k_current_site];

                    if (allele == 0) {
                        count_0++;
                    } else {
                        count_1++;
                    }
                }

                counts_0[tid + 1] = count_0;
                counts_1[tid + 1] = count_1;
            }

            // Bước 2: Scan prefix sum để xác định vị trí bắt đầu cho mỗi thread
            for (int t = 1; t <= actual_threads; ++t) {
                counts_0[t] += counts_0[t - 1];
                counts_1[t] += counts_1[t - 1];
            }

            // Bước 3: Mỗi thread điền dữ liệu vào vị trí của nó
            #pragma omp parallel num_threads(actual_threads)
            {
                const int tid = omp_get_thread_num();
                const int chunk_size = (M + actual_threads - 1) / actual_threads;
                const int start = tid * chunk_size;
                const int end = min(M, start + chunk_size);

                int offset_0 = counts_0[tid];
                int offset_1 = counts_1[tid];
                int p_div = k_current_site + 1;
                int q_div = k_current_site + 1;

                for (int i = start; i < end; ++i) {
                    p_div = max(p_div, d[i]);
                    q_div = max(q_div, d[i]);

                    const int current_hap = a[i];
                    const int allele = alleles[current_hap][k_current_site];

                    if (allele == 0) {
                        next_a[offset_0] = current_hap;
                        next_d[offset_0] = p_div;
                        offset_0++;
                        p_div = 0;
                    } else {
                        // Lưu vào phần sau của mảng
                        const int total_zeros = counts_0[actual_threads];
                        next_a[total_zeros + offset_1] = current_hap;
                        next_d[total_zeros + offset_1] = q_div;
                        offset_1++;
                        q_div = 0;
                    }
                }
            }
        } else {
            // Phương pháp tuần tự cho dữ liệu nhỏ
            vector<int> temp_b(M);
            vector<int> temp_e(M);

            int u_count = 0;
            int v_count = 0;
            int p_div = k_current_site + 1;
            int q_div = k_current_site + 1;

            for (int i = 0; i < M; ++i) {
                p_div = max(p_div, d[i]);
                q_div = max(q_div, d[i]);

                const int current_hap = a[i];
                const int allele = alleles[current_hap][k_current_site];

                if (allele == 0) {
                    next_a[u_count] = current_hap;
                    next_d[u_count] = p_div;
                    u_count++;
                    p_div = 0;
                } else {
                    temp_b[v_count] = current_hap;
                    temp_e[v_count] = q_div;
                    v_count++;
                    q_div = 0;
                }
            }

            if (v_count > 0) {
                memcpy(next_a.data() + u_count, temp_b.data(), v_count * sizeof(int));
                memcpy(next_d.data() + u_count, temp_e.data(), v_count * sizeof(int));
            }
        }

        // Swap thay vì copy
        a.swap(next_a);
        d.swap(next_d);
    }

    // Xử lý phần cuối cùng cho matches kết thúc tại N_sites
    if (N_sites > 0) {
        const int k_report_final = N_sites;
        int u_alg3_block_count0 = 0;
        int v_alg3_block_count1 = 0;
        const int last_site_idx = N_sites - 1;
        const int k_minus_L = k_report_final - L_param;

        for (int i_alg3_scan = 0; i_alg3_scan < M; ++i_alg3_scan) {
            const bool condition_block_break = (d[i_alg3_scan] > max(0, k_minus_L));

            if (condition_block_break) {
                if (u_alg3_block_count0 && v_alg3_block_count1) {
                    const int block_size = i_alg3_scan - i0_alg3_state;

                    if (block_size >= 1) {
                        // Sử dụng grain size linh hoạt
                        const int grain_size = max(1, min(16, block_size / (2 * actual_threads)));

                        #pragma omp parallel for schedule(dynamic, grain_size) num_threads(actual_threads)
                        for (int ia_loop = i0_alg3_state; ia_loop < i_alg3_scan; ++ia_loop) {
                            const int tid = omp_get_thread_num();
                            const int hap_ia = a[ia_loop];
                            const int allele_ia = alleles[hap_ia][last_site_idx];

                            int dmin_loop_private = 0;
                            for (int ib_loop = ia_loop + 1; ib_loop < i_alg3_scan; ++ib_loop) {
                                const int hap_ib = a[ib_loop];
                                dmin_loop_private = max(dmin_loop_private, d[ib_loop]);

                                if (alleles[hap_ib][last_site_idx] != allele_ia) {
                                    const int len = k_report_final - dmin_loop_private;
                                    if (len >= L_param) {
                                        thread_local_matches[tid].push_back({
                                            k_report_final,
                                            min(hap_ia, hap_ib),
                                            max(hap_ia, hap_ib),
                                            len
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                u_alg3_block_count0 = 0;
                v_alg3_block_count1 = 0;
                i0_alg3_state = i_alg3_scan;
            }

            if (alleles[a[i_alg3_scan]][last_site_idx] == 0) {
                u_alg3_block_count0++;
            } else {
                v_alg3_block_count1++;
            }
        }

        // Xử lý block cuối cùng
        if (u_alg3_block_count0 && v_alg3_block_count1) {
            const int block_size = M - i0_alg3_state;

            if (block_size >= 1) {
                // Sử dụng grain size linh hoạt
                const int grain_size = max(1, min(16, block_size / (2 * actual_threads)));

                #pragma omp parallel for schedule(dynamic, grain_size) num_threads(actual_threads)
                for (int ia_loop = i0_alg3_state; ia_loop < M; ++ia_loop) {
                    const int tid = omp_get_thread_num();
                    const int hap_ia = a[ia_loop];
                    const int allele_ia = alleles[hap_ia][last_site_idx];

                    int dmin_loop_private = 0;
                    for (int ib_loop = ia_loop + 1; ib_loop < M; ++ib_loop) {
                        const int hap_ib = a[ib_loop];
                        dmin_loop_private = max(dmin_loop_private, d[ib_loop]);

                        if (alleles[hap_ib][last_site_idx] != allele_ia) {
                            const int len = k_report_final - dmin_loop_private;
                            if (len >= L_param) {
                                thread_local_matches[tid].push_back({
                                    k_report_final,
                                    min(hap_ia, hap_ib),
                                    max(hap_ia, hap_ib),
                                    len
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Kết hợp kết quả từ tất cả các thread
    vector<vector<int>> all_matches;
    size_t total_matches = 0;

    // Tính tổng số matches để phân bổ trước
    for (const auto& tlm : thread_local_matches) {
        total_matches += tlm.size();
    }

    all_matches.reserve(total_matches);

    // Hiệu quả hơn khi sử dụng move thay vì copy
    for (auto& tlm : thread_local_matches) {
        if (!tlm.empty()) {
            all_matches.insert(all_matches.end(),
                               make_move_iterator(tlm.begin()),
                               make_move_iterator(tlm.end()));
        }
    }

    return all_matches;
}

int main(int argc, char *argv[]) {
    parallel_run(argc, argv);
}