/*
 * Copyright (c) 2026 morunner
 * * This file contains original code, as well as code and architectural
 * designs derived from https://github.com/marcneu/pcnhlslib
 * Original work Copyright (c) 2025 Marc Neu
 * * MIT License
 * * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef NNET_BITONIC_SORT_STREAM_H_
#define NNET_BITONIC_SORT_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_gravnet_bitonic_sort.h"
#include "nnet_types.h"

namespace nnet {

constexpr int log2(int x) { return (x <= 1) ? 0 : 1 + log2(x / 2); }

template <unsigned V, unsigned K, class dist_T, class idx_T>
void sort_node_array(hls::stream<nnet::array<dist_T, K>> &dist_in, hls::stream<nnet::array<idx_T, K>> &idx_in,
                     hls::stream<nnet::array<dist_T, K>> &dist_out, hls::stream<nnet::array<idx_T, K>> &idx_out) {
    for (unsigned int i = 0; i < V; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LATENCY min = 7 max = 8
        nnet::array<dist_T, K> d = dist_in.read();
        nnet::array<idx_T, K> idx = idx_in.read();

        bitonic_sort_array<K>(d, idx);

        dist_out.write(d);
        idx_out.write(idx);
    }
}

template <unsigned V, unsigned K, class dist_T, class idx_T>
void merge_node_arrays_and_keep_k(hls::stream<nnet::array<dist_T, K>> &left_dist_in,
                                  hls::stream<nnet::array<idx_T, K>> &left_idx_in,
                                  hls::stream<nnet::array<dist_T, K>> &right_dist_in,
                                  hls::stream<nnet::array<idx_T, K>> &right_idx_in,
                                  hls::stream<nnet::array<dist_T, K>> &dist_out,
                                  hls::stream<nnet::array<idx_T, K>> &idx_out) {
    for (unsigned int i = 0; i < V; i++) {
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

template <class dist_T, class idx_T, typename CONFIG_T>
void select_knn_tree(hls::stream<nnet::array<dist_T, CONFIG_T::n_neighbours>> dist_in[CONFIG_T::V / CONFIG_T::n_neighbours],
                     hls::stream<nnet::array<idx_T, CONFIG_T::n_neighbours>> idx_in[CONFIG_T::V / CONFIG_T::n_neighbours],
                     hls::stream<nnet::array<dist_T, CONFIG_T::n_neighbours>> &dist_out,
                     hls::stream<nnet::array<idx_T, CONFIG_T::n_neighbours>> &idx_out) {
#pragma HLS DATAFLOW

    constexpr int V = CONFIG_T::V;
    constexpr int K = CONFIG_T::n_neighbours;
    constexpr int num_leaves = CONFIG_T::V / K;
    constexpr int tree_depth = log2(num_leaves);

    hls::stream<nnet::array<dist_T, K>> dist_streams[tree_depth + 1][num_leaves];
    hls::stream<nnet::array<idx_T, K>> idx_streams[tree_depth + 1][num_leaves];
#pragma HLS STREAM variable = dist_streams depth = 2
#pragma HLS STREAM variable = idx_streams depth = 2

    for (int s = 0; s < tree_depth + 1; s++) {
#pragma HLS UNROLL
        if (s == 0) {
            for (int n = 0; n < num_leaves; n++) {
#pragma HLS UNROLL
                sort_node_array<V, K, dist_T, idx_T>(dist_in[n], idx_in[n], dist_streams[s][n], idx_streams[s][n]);
            }
        } else if (s < tree_depth) {
            int nodes_this_level = num_leaves >> s;
            for (int n = 0; n < nodes_this_level; n++) {
#pragma HLS UNROLL
                merge_node_arrays_and_keep_k<V, K, dist_T, idx_T>(
                    dist_streams[s - 1][2 * n], idx_streams[s - 1][2 * n], dist_streams[s - 1][2 * n + 1],
                    idx_streams[s - 1][2 * n + 1], dist_streams[s][n], idx_streams[s][n]);
            }
        } else {
            merge_node_arrays_and_keep_k<V, K, dist_T, idx_T>(dist_streams[s - 1][0], idx_streams[s - 1][0],
                                                              dist_streams[s - 1][1], idx_streams[s - 1][1], dist_out,
                                                              idx_out);
        }
    }
}

} // namespace nnet
#endif
