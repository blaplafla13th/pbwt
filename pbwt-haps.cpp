#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <cmath>
#include <limits>
#include <cstring> // For memcpy

#include "util.h"

using namespace std;

void core_p_arr(
    const vector<vector<int>>& haplotypes_sites_matrix,
    int site_idx,
    int n_hap,
    int num_threads,
    int block_size,
    const vector<int>& old_pda_parr,
    vector<int>& new_pda_parr,
    int& new_pda_zerocnt,
    vector<int>& ps_holder,
    vector<int>& offsets_holder)
{
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);

        if (s < e) {
            const int hap_s = old_pda_parr[s];
            ps_holder[s] = (haplotypes_sites_matrix[hap_s][site_idx] == 0) ? 0 : 1;

            for (int k = s + 1; k < e; ++k) {
                const int hap_k = old_pda_parr[k];
                ps_holder[k] = ps_holder[k - 1] + (haplotypes_sites_matrix[hap_k][site_idx] != 0);
            }
            offsets_holder[i] = ps_holder[e - 1];
        } else {
            offsets_holder[i] = 0;
        }
    }

    // Prefix sum on offsets
    for (int i = 1; i < num_threads; ++i) {
        offsets_holder[i] += offsets_holder[i - 1];
    }

    // Adjust ps values for each block
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);
        for (int k = s; k < e; ++k) {
            ps_holder[k] += offsets_holder[i - 1];
        }
    }

    // Calculate zerocnt and prepare for reordering
    if (n_hap > 0) {
        new_pda_zerocnt = n_hap - ps_holder[n_hap - 1];
    } else {
        new_pda_zerocnt = 0;
    }
    int one_off = new_pda_zerocnt - 1;

    // Reorder array based on bit values
    #pragma omp parallel for
    for (int i = 0; i < n_hap; ++i) {
        const int hap_i = old_pda_parr[i];
        if (haplotypes_sites_matrix[hap_i][site_idx] == 0) {
            new_pda_parr[i - ps_holder[i]] = hap_i;
        } else {
            new_pda_parr[ps_holder[i] + one_off] = hap_i;
        }
    }
}

void initial_sort(
    const vector<vector<int>>& haplotypes_sites_matrix,
    int site_idx,
    int n_hap,
    int num_threads,
    int block_size,
    vector<int>& old_pda_parr,
    vector<int>& old_pda_mlens,
    int& old_pda_zerocnt,
    vector<int>& ps_holder,
    vector<int>& offsets_holder)
{
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);

        if (s < e) {
            ps_holder[s] = (haplotypes_sites_matrix[s][site_idx] == 0) ? 0 : 1;

            for (int k = s + 1; k < e; ++k) {
                ps_holder[k] = ps_holder[k - 1] + (haplotypes_sites_matrix[k][site_idx] != 0);
            }
            offsets_holder[i] = ps_holder[e - 1];
        } else {
            offsets_holder[i] = 0;
        }
    }

    // Prefix sum on offsets
    for (int i = 1; i < num_threads; ++i) {
        offsets_holder[i] += offsets_holder[i - 1];
    }

    // Adjust ps values for each block
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);
        for (int k = s; k < e; ++k) {
            ps_holder[k] += offsets_holder[i - 1];
        }
    }

    // Calculate zerocnt and prepare for reordering
    if (n_hap > 0) {
        old_pda_zerocnt = n_hap - ps_holder[n_hap - 1];
    } else {
        old_pda_zerocnt = 0;
    }
    int one_off = old_pda_zerocnt - 1;

    // Initialize arrays
    #pragma omp parallel for
    for (int i = 0; i < n_hap; ++i) {
        if (haplotypes_sites_matrix[i][site_idx] == 0) {
            old_pda_parr[i - ps_holder[i]] = i;
            old_pda_mlens[i] = 1;
        } else {
            old_pda_parr[ps_holder[i] + one_off] = i;
            old_pda_mlens[i] = 1;
        }
    }

    // Initialize boundary values
    if (n_hap > 0) {
        if (!old_pda_parr.empty()) old_pda_mlens[old_pda_parr[0]] = 0;
        if (old_pda_zerocnt != n_hap) {
            if (old_pda_zerocnt >= 0 && old_pda_zerocnt < static_cast<int>(old_pda_parr.size())) {
                old_pda_mlens[old_pda_parr[old_pda_zerocnt]] = 0;
            }
        }
    }
}

void core_d_arr(
    const vector<vector<int>>& haplotypes_sites_matrix,
    int site_idx,
    int n_hap,
    int num_threads,
    int block_size,
    const vector<int>& old_pda_parr,
    const vector<int>& old_pda_mlens,
    vector<int>& new_pda_mlens,
    const vector<int>& ps_holder
)
{
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);

        int prv_low_m_zero = -1;
        int prv_low_m_one = -1;
        int h_id;

        if (s == 0) {
            for (int k = s; k < e; ++k) {
                h_id = old_pda_parr[k];
                if (haplotypes_sites_matrix[h_id][site_idx] == 0) {
                    new_pda_mlens[h_id] = min(old_pda_mlens[h_id], prv_low_m_one) + 1;
                    prv_low_m_one = numeric_limits<int>::max();
                    prv_low_m_zero = min(prv_low_m_zero, old_pda_mlens[h_id]);
                } else {
                    new_pda_mlens[h_id] = min(old_pda_mlens[h_id], prv_low_m_zero) + 1;
                    prv_low_m_zero = numeric_limits<int>::max();
                    prv_low_m_one = min(prv_low_m_one, old_pda_mlens[h_id]);
                }
            }
        } else {
            prv_low_m_zero = -1;
            prv_low_m_one = -1;

            bool min_zero_search = true;
            bool min_one_search = true;

            if (ps_holder[s - 1] == s){
                prv_low_m_zero = -1;
                min_zero_search = false;
            }
            if (ps_holder[s - 1] == 0) {
                prv_low_m_one = -1;
                min_one_search = false;
            }

            if (min_zero_search) {
                int seek_index = s - 1;
                while (seek_index >= 0 && haplotypes_sites_matrix[old_pda_parr[seek_index]][site_idx] != 0) {
                    seek_index--;
                }
                if (seek_index >= 0) {
                    prv_low_m_zero = old_pda_mlens[old_pda_parr[seek_index]];
                    while (seek_index >= 0 && haplotypes_sites_matrix[old_pda_parr[seek_index]][site_idx] == 0) {
                        prv_low_m_zero = min(old_pda_mlens[old_pda_parr[seek_index]], prv_low_m_zero);
                        seek_index--;
                    }
                } else {
                    prv_low_m_zero = -1;
                }
            }

            if (min_one_search) {
                int seek_index = s - 1;
                while (seek_index >= 0 && haplotypes_sites_matrix[old_pda_parr[seek_index]][site_idx] == 0) {
                    seek_index--;
                }
                if (seek_index >=0) {
                    prv_low_m_one = old_pda_mlens[old_pda_parr[seek_index]];
                    while (seek_index >= 0 && haplotypes_sites_matrix[old_pda_parr[seek_index]][site_idx] != 0) {
                        prv_low_m_one = min(old_pda_mlens[old_pda_parr[seek_index]], prv_low_m_one);
                        seek_index--;
                    }
                } else {
                    prv_low_m_one = -1;
                }
            }

            for (int k = s; k < e; ++k) {
                h_id = old_pda_parr[k];
                if (haplotypes_sites_matrix[h_id][site_idx] == 0) {
                    new_pda_mlens[h_id] = min(old_pda_mlens[h_id], prv_low_m_one) + 1;
                    prv_low_m_one = numeric_limits<int>::max();
                    prv_low_m_zero = min(prv_low_m_zero, old_pda_mlens[h_id]);
                } else {
                    new_pda_mlens[h_id] = min(old_pda_mlens[h_id], prv_low_m_zero) + 1;
                    prv_low_m_zero = numeric_limits<int>::max();
                    prv_low_m_one = min(prv_low_m_one, old_pda_mlens[h_id]);
                }
            }
        }
    }
}

vector<vector<int>> report_long_matches_sl(
    int site_index_0_based,
    const vector<vector<int>>& haplotypes_sites_matrix,
    int site_idx,
    int n_hap,
    int num_threads,
    int llm_len,
    const vector<int>& old_pda_parr,
    const vector<int>& old_pda_mlens)
{
    // Tối ưu bằng cách dự đoán kích thước
    const int max_expected_matches = min(n_hap * n_hap / 4, 100000);
    vector<vector<vector<int>>> per_thread_matches_storage(num_threads);
    for (auto& thread_storage : per_thread_matches_storage) {
        thread_storage.reserve(max_expected_matches / num_threads);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 32)
        for (int h = 0; h < n_hap - 1; ++h) {
            int min_val = numeric_limits<int>::max();
            const int hap_h = old_pda_parr[h];
            const int allele_h = haplotypes_sites_matrix[hap_h][site_idx];

            for (int i = h + 1; i < n_hap; ++i) {
                const int hap_i = old_pda_parr[i];
                if (old_pda_mlens[hap_i] < llm_len) {
                    break;
                }
                min_val = min(min_val, old_pda_mlens[hap_i]);

                if (haplotypes_sites_matrix[hap_i][site_idx] != allele_h) {
                    per_thread_matches_storage[thread_id].push_back({
                        site_index_0_based + 1,
                        min(hap_h, hap_i),
                        max(hap_h, hap_i),
                        min_val
                    });
                }
            }
        }
    }

    // Tính toán tổng số matches để dự đoán kích thước
    size_t total_matches = 0;
    for (const auto& thread_list : per_thread_matches_storage) {
        total_matches += thread_list.size();
    }

    vector<vector<int>> aggregated_function_matches;
    aggregated_function_matches.reserve(total_matches);

    // Sử dụng move iterator để tối ưu việc nối
    for (auto& thread_list : per_thread_matches_storage) {
        if (!thread_list.empty()) {
            aggregated_function_matches.insert(
                aggregated_function_matches.end(),
                make_move_iterator(thread_list.begin()),
                make_move_iterator(thread_list.end())
            );
        }
    }

    return aggregated_function_matches;
}

vector<vector<int>> report_long_matches_sl_tail(
    int site_index_0_based,
    int n_hap,
    int num_threads,
    int llm_len,
    const vector<int>& old_pda_parr,
    const vector<int>& old_pda_mlens)
{
    // Tối ưu bằng cách dự đoán kích thước
    const int max_expected_matches = min(n_hap * n_hap / 4, 100000);
    vector<vector<vector<int>>> per_thread_matches_storage(num_threads);
    for (auto& thread_storage : per_thread_matches_storage) {
        thread_storage.reserve(max_expected_matches / num_threads);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 32)
        for (int h = 0; h < n_hap - 1; ++h) {
            int min_val = numeric_limits<int>::max();
            for (int i = h + 1; i < n_hap; ++i) {
                if (old_pda_mlens[old_pda_parr[i]] < llm_len) {
                    break;
                }
                min_val = min(min_val, old_pda_mlens[old_pda_parr[i]]);
                per_thread_matches_storage[thread_id].push_back({
                    site_index_0_based + 1,
                    min(old_pda_parr[h], old_pda_parr[i]),
                    max(old_pda_parr[h], old_pda_parr[i]),
                    min_val
                });
            }
        }
    }

    // Tính toán tổng số matches để dự đoán kích thước
    size_t total_matches = 0;
    for (const auto& thread_list : per_thread_matches_storage) {
        total_matches += thread_list.size();
    }

    vector<vector<int>> aggregated_function_matches;
    aggregated_function_matches.reserve(total_matches);

    // Sử dụng move iterator để tối ưu việc nối
    for (auto& thread_list : per_thread_matches_storage) {
        if (!thread_list.empty()) {
            aggregated_function_matches.insert(
                aggregated_function_matches.end(),
                make_move_iterator(thread_list.begin()),
                make_move_iterator(thread_list.end())
            );
        }
    }

    return aggregated_function_matches;
}

vector<vector<int>> build_and_match(
    const vector<vector<int>>& haplotypes_sites_matrix,
    int num_threads,
    int L_long_match_min_len)
{
    if (haplotypes_sites_matrix.empty() || haplotypes_sites_matrix[0].empty()) {
        return {};
    }

    const int n_hap = haplotypes_sites_matrix.size();
    const int n_sites = haplotypes_sites_matrix[0].size();

    int block_size = (n_hap + num_threads - 1) / num_threads;
    if (num_threads <= 0) block_size = n_hap;

    // Khởi tạo các vector với kích thước cố định
    vector<int> new_pda_parr(n_hap);
    vector<int> new_pda_mlens(n_hap);
    int new_pda_zerocnt = 0;

    vector<int> old_pda_parr(n_hap);
    vector<int> old_pda_mlens(n_hap);
    int old_pda_zerocnt = 0;

    vector<int> ps_holder(n_hap);
    vector<int> offsets_holder(max(1, num_threads));

    // Dự đoán kích thước cho all_matches_report
    vector<vector<int>> all_matches_report;
    all_matches_report.reserve(min(n_hap * n_sites / 10, 1000000));

    // Khởi tạo sắp xếp ban đầu cho site đầu tiên
    initial_sort(haplotypes_sites_matrix, 0, n_hap, num_threads, block_size,
                old_pda_parr, old_pda_mlens, old_pda_zerocnt,
                ps_holder, offsets_holder);

    for (int site_idx = 1; site_idx < n_sites; ++site_idx) {
        // Xây dựng mảng p mới
        core_p_arr(haplotypes_sites_matrix, site_idx, n_hap, num_threads, block_size,
                  old_pda_parr, new_pda_parr, new_pda_zerocnt,
                  ps_holder, offsets_holder);

        // Xây dựng mảng d mới
        core_d_arr(haplotypes_sites_matrix, site_idx, n_hap, num_threads, block_size,
                  old_pda_parr, old_pda_mlens, new_pda_mlens, ps_holder);

        // Báo cáo các chuỗi khớp dài
        vector<vector<int>> current_site_matches = report_long_matches_sl(
            site_idx - 1, haplotypes_sites_matrix, site_idx, n_hap, num_threads,
            L_long_match_min_len, old_pda_parr, old_pda_mlens);

        // Nối kết quả
        if (!current_site_matches.empty()) {
            all_matches_report.insert(
                all_matches_report.end(),
                make_move_iterator(current_site_matches.begin()),
                make_move_iterator(current_site_matches.end())
            );
        }

        // Hoán đổi các mảng cũ và mới
        old_pda_parr.swap(new_pda_parr);
        old_pda_mlens.swap(new_pda_mlens);
        old_pda_zerocnt = new_pda_zerocnt;
    }

    // Xử lý site cuối cùng
    if (n_sites > 0) {
        vector<vector<int>> tail_matches = report_long_matches_sl_tail(
            n_sites - 1, n_hap, num_threads, L_long_match_min_len,
            old_pda_parr, old_pda_mlens);

        // Nối kết quả
        if (!tail_matches.empty()) {
            all_matches_report.insert(
                all_matches_report.end(),
                make_move_iterator(tail_matches.begin()),
                make_move_iterator(tail_matches.end())
            );
        }
    }

    return all_matches_report;
}

int main(int argc, char *argv[]) {
    parallel_run(argc, argv);
}