#ifndef NNET_BITONIC_SORT_STREAM_H_
#define NNET_BITONIC_SORT_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_gravnet_bitonic_sort.h"
#include "nnet_types.h"

namespace nnet {

constexpr int exact_log2(int x) { return (x <= 1) ? 0 : 1 + exact_log2(x / 2); }

template <unsigned n_iterations, unsigned K, class dist_T, class idx_T>
void sort_chunk_node(hls::stream<nnet::array<dist_T, K>> &dist_in, hls::stream<nnet::array<idx_T, K>> &idx_in,
                     hls::stream<nnet::array<dist_T, K>> &dist_out, hls::stream<nnet::array<idx_T, K>> &idx_out) {
    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1
        nnet::array<dist_T, K> d = dist_in.read();
        nnet::array<idx_T, K> idx = idx_in.read();

        bitonic_sort_array<K>(d, idx);

        dist_out.write(d);
        idx_out.write(idx);
    }
}

template <unsigned n_iterations, unsigned K, class dist_T, class idx_T>
void merge_chunk_node(hls::stream<nnet::array<dist_T, K>> &left_dist_in, hls::stream<nnet::array<idx_T, K>> &left_idx_in,
                      hls::stream<nnet::array<dist_T, K>> &right_dist_in, hls::stream<nnet::array<idx_T, K>> &right_idx_in,
                      hls::stream<nnet::array<dist_T, K>> &dist_out, hls::stream<nnet::array<idx_T, K>> &idx_out) {
    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1
        nnet::array<dist_T, K> left_d = left_dist_in.read();
        nnet::array<idx_T, K> left_i = left_idx_in.read();
        nnet::array<dist_T, K> right_d = right_dist_in.read();
        nnet::array<idx_T, K> right_i = right_idx_in.read();

        nnet::array<dist_T, K> merged_d;
        nnet::array<idx_T, K> merged_i;

        merge_and_keep_k<K>(left_d, left_i, right_d, right_i, merged_d, merged_i);

        dist_out.write(merged_d);
        idx_out.write(merged_i);
    }
}

template <unsigned n_iterations, unsigned n_pack, class dist_T, class idx_T, typename CONFIG_T>
void select_knn_tree(
    hls::stream<nnet::array<dist_T, CONFIG_T::n_neighbours>> dist_in[n_pack][CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<idx_T, CONFIG_T::n_neighbours>> idx_in[n_pack][CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<dist_T, CONFIG_T::n_neighbours>> dist_out[n_pack],
    hls::stream<nnet::array<idx_T, CONFIG_T::n_neighbours>> idx_out[n_pack]) {
#pragma HLS DATAFLOW

    constexpr int K = CONFIG_T::n_neighbours;
    constexpr int num_leaves = CONFIG_T::V / K;
    constexpr int S = exact_log2(num_leaves);

    hls::stream<nnet::array<dist_T, K>> dist_streams[S + 1][n_pack][num_leaves];
    hls::stream<nnet::array<idx_T, K>> idx_streams[S + 1][n_pack][num_leaves];
#pragma HLS STREAM variable = dist_streams depth = 2
#pragma HLS STREAM variable = idx_streams depth = 2

    for (int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL
        for (int s = 0; s < S + 1; s++) {
#pragma HLS UNROLL
            if (s == 0) {
                for (int n = 0; n < num_leaves; n++) {
#pragma HLS UNROLL
                    sort_chunk_node<n_iterations, K, dist_T, idx_T>(dist_in[p][n], idx_in[p][n], dist_streams[s][p][n],
                                                                    idx_streams[s][p][n]);
                }
            } else if (s < S) {
                int nodes_this_level = num_leaves >> s;
                for (int n = 0; n < nodes_this_level; n++) {
#pragma HLS UNROLL
                    merge_chunk_node<n_iterations, K, dist_T, idx_T>(
                        dist_streams[s - 1][p][2 * n], idx_streams[s - 1][p][2 * n], dist_streams[s - 1][p][2 * n + 1],
                        idx_streams[s - 1][p][2 * n + 1], dist_streams[s][p][n], idx_streams[s][p][n]);
                }
            } else {
                merge_chunk_node<n_iterations, K, dist_T, idx_T>(dist_streams[s - 1][p][0], idx_streams[s - 1][p][0],
                                                                 dist_streams[s - 1][p][1], idx_streams[s - 1][p][1],
                                                                 dist_out[p], idx_out[p]);
            }
        }
    }
}

} // namespace nnet
#endif
