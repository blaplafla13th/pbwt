#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <omp.h>
#include <cmath>
#include <limits>

#include "util.h"

using namespace std;

void core_p_arr(
    const function<char(int)>& get_hap_val,
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
            if (get_hap_val(old_pda_parr[s]) == '0') {
                ps_holder[s] = 0;
            } else {
                ps_holder[s] = 1;
            }

            for (int k = s + 1; k < e; ++k) {
                if (get_hap_val(old_pda_parr[k]) == '0') {
                    ps_holder[k] = ps_holder[k - 1];
                } else {
                    ps_holder[k] = ps_holder[k - 1] + 1;
                }
            }
            offsets_holder[i] = ps_holder[e - 1];
        } else {
             offsets_holder[i] = 0;
        }
    }

    for (int i = 1; i < num_threads; ++i) {
        offsets_holder[i] += offsets_holder[i - 1];
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);
        for (int k = s; k < e; ++k) {
            ps_holder[k] += offsets_holder[i - 1];
        }
    }

    if (n_hap > 0) {
        new_pda_zerocnt = n_hap - ps_holder[n_hap - 1];
    } else {
        new_pda_zerocnt = 0;
    }
    int one_off = new_pda_zerocnt -1;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < n_hap; ++i) {
        if (get_hap_val(old_pda_parr[i]) == '0') {
            new_pda_parr[i - ps_holder[i]] = old_pda_parr[i];
        } else {
            new_pda_parr[ps_holder[i] + one_off] = old_pda_parr[i];
        }
    }
}

void initial_sort(
    const function<char(int)>& get_hap_val_initial,
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
            if (get_hap_val_initial(s) == '0') {
                ps_holder[s] = 0;
            } else {
                ps_holder[s] = 1;
            }

            for (int k = s + 1; k < e; ++k) {
                if (get_hap_val_initial(k) == '0') {
                    ps_holder[k] = ps_holder[k - 1];
                } else {
                    ps_holder[k] = ps_holder[k - 1] + 1;
                }
            }
            offsets_holder[i] = ps_holder[e - 1];
        } else {
            offsets_holder[i] = 0;
        }
    }

    for (int i = 1; i < num_threads; ++i) {
        offsets_holder[i] += offsets_holder[i - 1];
    }

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 1; i < num_threads; ++i) {
        int s = i * block_size;
        if (s >= n_hap) continue;
        int e = min(s + block_size, n_hap);
        for (int k = s; k < e; ++k) {
            ps_holder[k] += offsets_holder[i - 1];
        }
    }

    if (n_hap > 0) {
        old_pda_zerocnt = n_hap - ps_holder[n_hap - 1];
    } else {
        old_pda_zerocnt = 0;
    }
    int one_off = old_pda_zerocnt -1;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for
    for (int i = 0; i < n_hap; ++i) {
        if (get_hap_val_initial(i) == '0') {
            old_pda_parr[i - ps_holder[i]] = i;
            old_pda_mlens[i] = 1;
        } else {
            old_pda_parr[ps_holder[i] + one_off] = i;
            old_pda_mlens[i] = 1;
        }
    }

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
    const function<char(int)>& get_hap_val,
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
                if (get_hap_val(h_id) == '0') {
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
                while (seek_index >= 0 && get_hap_val(old_pda_parr[seek_index]) != '0') {
                    seek_index--;
                }
                if (seek_index >= 0) {
                    prv_low_m_zero = old_pda_mlens[old_pda_parr[seek_index]];
                    while (seek_index >= 0 && get_hap_val(old_pda_parr[seek_index]) == '0') {
                        prv_low_m_zero = min(old_pda_mlens[old_pda_parr[seek_index]], prv_low_m_zero);
                        seek_index--;
                    }
                } else {
                    prv_low_m_zero = -1;
                }
            }

            if (min_one_search) {
                int seek_index = s - 1;
                while (seek_index >= 0 && get_hap_val(old_pda_parr[seek_index]) == '0') {
                    seek_index--;
                }
                if (seek_index >=0) {
                    prv_low_m_one = old_pda_mlens[old_pda_parr[seek_index]];
                    while (seek_index >= 0 && get_hap_val(old_pda_parr[seek_index]) != '0') {
                        prv_low_m_one = min(old_pda_mlens[old_pda_parr[seek_index]], prv_low_m_one);
                        seek_index--;
                    }
                } else {
                    prv_low_m_one = -1;
                }
            }

            for (int k = s; k < e; ++k) {
                h_id = old_pda_parr[k];
                if (get_hap_val(h_id) == '0') {
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

// Hàm này giờ trả về vector kết quả của nó
vector<vector<int>> report_long_matches_sl(
    int site_index_0_based,
    const function<char(int)>& get_hap_val,
    int n_hap,
    int num_threads,
    int llm_len,
    const vector<int>& old_pda_parr,
    const vector<int>& old_pda_mlens)
{
    vector<vector<vector<int>>> per_thread_matches_storage(num_threads);

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int h = 0; h < n_hap - 1; ++h) {
            int min_val = numeric_limits<int>::max();
            for (int i = h + 1; i < n_hap; ++i) {
                if (old_pda_mlens[old_pda_parr[i]] < llm_len) {
                    break;
                }
                min_val = min(min_val, old_pda_mlens[old_pda_parr[i]]);

                if (get_hap_val(old_pda_parr[h]) != get_hap_val(old_pda_parr[i])) {
                    per_thread_matches_storage[thread_id].push_back({site_index_0_based + 1,
                                                                    min(old_pda_parr[h], old_pda_parr[i]),
                                                                    max(old_pda_parr[h], old_pda_parr[i]),
                                                                    min_val});
                }
            }
        }
    }

    vector<vector<int>> aggregated_function_matches;
    // Gộp kết quả từ các vector của từng luồng
    for(const auto& thread_list : per_thread_matches_storage) {
        aggregated_function_matches.insert(aggregated_function_matches.end(), thread_list.begin(), thread_list.end());
    }
    return aggregated_function_matches;
}

// Hàm này giờ trả về vector kết quả của nó
vector<vector<int>> report_long_matches_sl_tail(
    int site_index_0_based,
    int n_hap,
    int num_threads,
    int llm_len,
    const vector<int>& old_pda_parr,
    const vector<int>& old_pda_mlens)
{
    vector<vector<vector<int>>> per_thread_matches_storage(num_threads);

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic)
        for (int h = 0; h < n_hap - 1; ++h) {
            int min_val = numeric_limits<int>::max();
            for (int i = h + 1; i < n_hap; ++i) {
                if (old_pda_mlens[old_pda_parr[i]] < llm_len) {
                     break;
                }
                min_val = min(min_val, old_pda_mlens[old_pda_parr[i]]);
                per_thread_matches_storage[thread_id].push_back({site_index_0_based + 1,
                                                               min(old_pda_parr[h], old_pda_parr[i]),
                                                               max(old_pda_parr[h], old_pda_parr[i]),
                                                               min_val});
            }
        }
    }

    vector<vector<int>> aggregated_function_matches;
    // Gộp kết quả từ các vector của từng luồng
    for(const auto& thread_list : per_thread_matches_storage) {
        aggregated_function_matches.insert(aggregated_function_matches.end(), thread_list.begin(), thread_list.end());
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
    int n_hap = haplotypes_sites_matrix.size();
    int n_sites = haplotypes_sites_matrix[0].size();

    int block_size = (n_hap + num_threads - 1) / num_threads;
    if (num_threads <= 0) block_size = n_hap;

    vector<int> new_pda_parr(n_hap);
    vector<int> new_pda_mlens(n_hap);
    int new_pda_zerocnt = 0;

    vector<int> old_pda_parr(n_hap);
    vector<int> old_pda_mlens(n_hap);
    int old_pda_zerocnt = 0;

    vector<int> ps_holder(n_hap);
    vector<int> offsets_holder(max(1,num_threads));

    vector<vector<int>> all_matches_report; // Final output: {site_1_based, hap1, hap2, length}

    auto get_hap_val_for_site_k =
        [&](int current_processing_site_idx, int original_hap_id) -> char {
        return haplotypes_sites_matrix[original_hap_id][current_processing_site_idx] == 0 ? '0' : '1';
    };

    initial_sort(
        [&](int original_hap_id){ return get_hap_val_for_site_k(0, original_hap_id); },
        n_hap, num_threads, block_size,
        old_pda_parr, old_pda_mlens, old_pda_zerocnt,
        ps_holder, offsets_holder);

    for (int site_idx = 1; site_idx < n_sites; ++site_idx) { // site_idx is 0-based index of current site
        core_p_arr(
            [&](int original_hap_id){ return get_hap_val_for_site_k(site_idx, original_hap_id); },
            n_hap, num_threads, block_size,
            old_pda_parr,
            new_pda_parr, new_pda_zerocnt,
            ps_holder, offsets_holder);

        core_d_arr(
            [&](int original_hap_id){ return get_hap_val_for_site_k(site_idx, original_hap_id); },
            n_hap, num_threads, block_size,
            old_pda_parr,
            old_pda_mlens,
            new_pda_mlens,
            ps_holder
        );

        vector<vector<int>> current_site_matches = report_long_matches_sl(
            site_idx - 1, // 0-based site index where match ends
            [&](int original_hap_id){ return get_hap_val_for_site_k(site_idx, original_hap_id); },
            n_hap, num_threads, L_long_match_min_len,
            old_pda_parr,
            old_pda_mlens);

        // Nối kết quả vào all_matches_report. Vì vòng lặp site_idx là tuần tự, không cần critical.
        // Sử dụng reserve để tối ưu hóa việc cấp phát bộ nhớ nếu biết trước kích thước gần đúng.
        if (!current_site_matches.empty()) {
            if (all_matches_report.capacity() < all_matches_report.size() + current_site_matches.size()) {
                 all_matches_report.reserve(all_matches_report.size() + current_site_matches.size()); // Ước tính
            }
            all_matches_report.insert(all_matches_report.end(), current_site_matches.begin(), current_site_matches.end());
        }


        old_pda_parr.swap(new_pda_parr);
        old_pda_mlens.swap(new_pda_mlens);
        old_pda_zerocnt = new_pda_zerocnt;
    }

    if (n_sites > 0) {
        vector<vector<int>> tail_matches = report_long_matches_sl_tail(
            n_sites - 1, // 0-based index of the last site
            n_hap, num_threads, L_long_match_min_len,
            old_pda_parr,
            old_pda_mlens);

        // Nối kết quả vào all_matches_report.
        if (!tail_matches.empty()) {
            if (all_matches_report.capacity() < all_matches_report.size() + tail_matches.size()) {
                 all_matches_report.reserve(all_matches_report.size() + tail_matches.size()); // Ước tính
            }
            all_matches_report.insert(all_matches_report.end(), tail_matches.begin(), tail_matches.end());
        }
    }

    return all_matches_report;
}

int main(int argc, char *argv[]) {
    parallel_run(argc, argv);
}
