#include <vector>
#include <numeric> // For iota
#include <algorithm> // For copy, min, max
#include <omp.h> // For OpenMP

#include "util.h"

using namespace std;

vector<vector<int>> build_and_match(const vector<vector<int>>& hap_map_rows_haps, int num_threads, int L_param) {
    const int M = hap_map_rows_haps.size();
    if (M == 0) return {};
    const int N_sites = hap_map_rows_haps[0].size();
    if (N_sites == 0) return {};

    vector<vector<int>> all_matches;

    vector<int> a(M);
    iota(a.begin(), a.end(), 0);
    vector<int> d(M, 0);

    int i0_alg3_state = 0;
    int L_param_int = (int)L_param;

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    for (int k_current_site = 0; k_current_site < N_sites; ++k_current_site) {
        // Inlined Algorithm 3: ReportLongMatches (for matches ending before or at k_current_site)
        {
            int u_alg3_block_count0 = 0;
            int v_alg3_block_count1 = 0;

            for (int i_alg3_scan = 0; i_alg3_scan < M; ++i_alg3_scan) {
                bool condition_block_break;
                if (k_current_site < L_param_int) {
                    condition_block_break = ((long long)d[i_alg3_scan] > (long long)k_current_site - (long long)L_param_int);
                } else {
                    condition_block_break = (d[i_alg3_scan] > (k_current_site - L_param_int));
                }

                if (condition_block_break) {
                    if (u_alg3_block_count0 && v_alg3_block_count1) {
                        #pragma omp parallel for schedule(dynamic)
                        for (int ia_loop = i0_alg3_state; ia_loop < i_alg3_scan; ++ia_loop) {
                            int dmin_loop_private = 0;
                            for (int ib_loop = ia_loop + 1; ib_loop < i_alg3_scan; ++ib_loop) {
                                if (d[ib_loop] > dmin_loop_private) dmin_loop_private = d[ib_loop];
                                if (hap_map_rows_haps[a[ib_loop]][k_current_site] != hap_map_rows_haps[a[ia_loop]][k_current_site]) {
                                    if (k_current_site > dmin_loop_private) {
                                        int len = k_current_site - dmin_loop_private;
                                        if (len >= L_param_int) {
                                            int hap1 = min(a[ia_loop], a[ib_loop]);
                                            int hap2 = max(a[ia_loop], a[ib_loop]);
                                            #pragma omp critical
                                            {
                                                all_matches.push_back({k_current_site, (int)hap1, (int)hap2, (int)len});
                                            }
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
                if (hap_map_rows_haps[a[i_alg3_scan]][k_current_site] == 0) {
                    u_alg3_block_count0++;
                } else {
                    v_alg3_block_count1++;
                }
            }
            // Process the last block after the loop for current k_current_site
            if (u_alg3_block_count0 && v_alg3_block_count1) {
                 #pragma omp parallel for schedule(dynamic)
                 for (int ia_loop = i0_alg3_state; ia_loop < M; ++ia_loop) {
                    int dmin_loop_private = 0;
                    for (int ib_loop = ia_loop + 1; ib_loop < M; ++ib_loop) {
                        if (d[ib_loop] > dmin_loop_private) dmin_loop_private = d[ib_loop];
                         if (hap_map_rows_haps[a[ib_loop]][k_current_site] != hap_map_rows_haps[a[ia_loop]][k_current_site]) {
                            if (k_current_site > dmin_loop_private) {
                                int len = k_current_site - dmin_loop_private;
                                if (len >= L_param_int) {
                                    int hap1 = min(a[ia_loop], a[ib_loop]);
                                    int hap2 = max(a[ia_loop], a[ib_loop]);
                                    #pragma omp critical
                                    {
                                        all_matches.push_back({k_current_site, (int)hap1, (int)hap2, (int)len});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Inlined Algorithm 2: BuildPrefixAndDivergenceArrays (updates a, d for next site)
        {
            vector<int> next_a(M);
            vector<int> next_d(M);
            vector<int> temp_b_for_alg2(M);
            vector<int> temp_e_for_alg2(M);

            int u_alg2_fill_count = 0;
            int v_alg2_fill_count = 0;
            int p_alg2_div = k_current_site + 1;
            int q_alg2_div = k_current_site + 1;

            for (int i_alg2_scan = 0; i_alg2_scan < M; ++i_alg2_scan) {
                 if (d[i_alg2_scan] > p_alg2_div) p_alg2_div = d[i_alg2_scan];
                 if (d[i_alg2_scan] > q_alg2_div) q_alg2_div = d[i_alg2_scan];

                 if (hap_map_rows_haps[a[i_alg2_scan]][k_current_site] == 0) {
                    next_a[u_alg2_fill_count] = a[i_alg2_scan];
                    next_d[u_alg2_fill_count] = p_alg2_div;
                    u_alg2_fill_count++;
                    p_alg2_div = 0;
                 } else {
                    temp_b_for_alg2[v_alg2_fill_count] = a[i_alg2_scan];
                    temp_e_for_alg2[v_alg2_fill_count] = q_alg2_div;
                    v_alg2_fill_count++;
                    q_alg2_div = 0;
                 }
            }
            copy(temp_b_for_alg2.begin(), temp_b_for_alg2.begin() + v_alg2_fill_count, next_a.begin() + u_alg2_fill_count);
            copy(temp_e_for_alg2.begin(), temp_e_for_alg2.begin() + v_alg2_fill_count, next_d.begin() + u_alg2_fill_count);

            a = next_a;
            d = next_d;
        }
    }

    // Final reporting for matches ending at N_sites (i.e. up to site N_sites-1)
    // This uses a and d computed after processing site N_sites-1
    // The k_report_val is N_sites (exclusive end of match interval)
    // Alleles are from site N_sites-1
    if (N_sites > 0) { // Only if there are sites
        const int k_report_final = N_sites;
        int u_alg3_block_count0 = 0;
        int v_alg3_block_count1 = 0;

        for (int i_alg3_scan = 0; i_alg3_scan < M; ++i_alg3_scan) {
            bool condition_block_break;
            if (k_report_final < L_param_int) {
                 condition_block_break = ((long long)d[i_alg3_scan] > (long long)k_report_final - (long long)L_param_int);
            } else {
                 condition_block_break = (d[i_alg3_scan] > (k_report_final - L_param_int));
            }

            if (condition_block_break) {
                if (u_alg3_block_count0 && v_alg3_block_count1) {
                    #pragma omp parallel for schedule(dynamic)
                    for (int ia_loop = i0_alg3_state; ia_loop < i_alg3_scan; ++ia_loop) {
                        int dmin_loop_private = 0;
                        for (int ib_loop = ia_loop + 1; ib_loop < i_alg3_scan; ++ib_loop) {
                            if (d[ib_loop] > dmin_loop_private) dmin_loop_private = d[ib_loop];
                            if (hap_map_rows_haps[a[ib_loop]][N_sites-1] != hap_map_rows_haps[a[ia_loop]][N_sites-1]) {
                                if (k_report_final > dmin_loop_private) {
                                    int len = k_report_final - dmin_loop_private;
                                    if (len >= L_param_int) {
                                        int hap1 = min(a[ia_loop], a[ib_loop]);
                                        int hap2 = max(a[ia_loop], a[ib_loop]);
                                        #pragma omp critical
                                        {
                                            all_matches.push_back({k_report_final, (int)hap1, (int)hap2, (int)len});
                                        }
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
            if (hap_map_rows_haps[a[i_alg3_scan]][N_sites-1] == 0) {
                u_alg3_block_count0++;
            } else {
                v_alg3_block_count1++;
            }
        }
        // Process the last block for k_report_final
        if (u_alg3_block_count0 && v_alg3_block_count1) {
            #pragma omp parallel for schedule(dynamic)
            for (int ia_loop = i0_alg3_state; ia_loop < M; ++ia_loop) {
                int dmin_loop_private = 0;
                for (int ib_loop = ia_loop + 1; ib_loop < M; ++ib_loop) {
                    if (d[ib_loop] > dmin_loop_private) dmin_loop_private = d[ib_loop];
                    if (hap_map_rows_haps[a[ib_loop]][N_sites-1] != hap_map_rows_haps[a[ia_loop]][N_sites-1]) {
                        if (k_report_final > dmin_loop_private) {
                            int len = k_report_final - dmin_loop_private;
                            if (len >= L_param_int) {
                                int hap1 = min(a[ia_loop], a[ib_loop]);
                                int hap2 = max(a[ia_loop], a[ib_loop]);
                                #pragma omp critical
                                {
                                   all_matches.push_back({k_report_final, (int)hap1, (int)hap2, (int)len});
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return all_matches;
}

int main(int argc, char *argv[]) {
    parallel_run(argc, argv);
}
