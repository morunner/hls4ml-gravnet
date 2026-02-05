#ifndef NNET_GRAVNET_BITONIC_SORT_STREAM_H_
#define NNET_GRAVNET_BITONIC_SORT_STREAM_H_

#include "hls_stream.h"
#include "nnet_types.h"
#include <algorithm>

namespace nnet {

/**
 * @brief Swaps two given pairs (distance, index)
 */
template <typename dist_T, typename idx_T> void gravnet_swap_stream(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    dist_T temp_d = d1;
    d1 = d2;
    d2 = temp_d;
    idx_T temp_i = i1;
    i1 = i2;
    i2 = temp_i;
}

/**
 * @brief Compares two elements (distance, index) and assigns the pair with the smaller distance to the first element
 */
template <typename dist_T, typename idx_T>
void gravnet_compare_and_swap_stream(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    if (d2 < d1) {
        gravnet_swap_stream(d1, i1, d2, i2);
    }
}

/**
 * @brief Sorts a bitonic sequence by dist and keeps track of the corresponding index idx
 */
template <int N, typename dist_T, typename idx_T>
void sort_bitonic_sequence_stream(nnet::array<dist_T, N> &dist, nnet::array<idx_T, N> &idx) {
#pragma HLS INLINE
    for (int stride = N / 2; stride > 0; stride /= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            if ((i % (2 * stride)) < stride) {
                gravnet_compare_and_swap_stream(dist[i], idx[i], dist[i + stride], idx[i + stride]);
            }
        }
    }
}

/**
 * @brief Sorts a given array using bitonic sort and keeps track of the corresponding index
 */
template <int N, typename dist_T, typename idx_T>
void bitonic_sort_array_stream(nnet::array<dist_T, N> &dist, nnet::array<idx_T, N> &idx) {
#pragma HLS INLINE
    for (int seq_size = 2; seq_size <= N; seq_size *= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < N; i += seq_size) {
#pragma HLS UNROLL

            int start = i;
            int end = i + seq_size - 1;
            for (int k = 0; k < seq_size / 2; k++) {
#pragma HLS UNROLL
                int idx1 = start + k;
                int idx2 = end - k;
                gravnet_compare_and_swap_stream(dist[idx1], idx[idx1], dist[idx2], idx[idx2]);
            }

            for (int stride = seq_size / 4; stride > 0; stride /= 2) {
#pragma HLS UNROLL
                for (int j = 0; j < seq_size; j++) {
#pragma HLS UNROLL
                    if ((j % (2 * stride)) < stride) {
                        int pos = i + j;
                        gravnet_compare_and_swap_stream(dist[pos], idx[pos], dist[pos + stride], idx[pos + stride]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Bitonically merges two sorted lists and keeps the top-k elements
 */
template <int K, typename dist_T, typename idx_T>
void merge_and_keep_k_stream(nnet::array<dist_T, K> &left_dists, nnet::array<idx_T, K> &left_indices,
                             nnet::array<dist_T, K> &right_dists, nnet::array<idx_T, K> &right_indices,
                             nnet::array<dist_T, K> &dist_out, nnet::array<idx_T, K> &idx_out) {
#pragma HLS INLINE

    // We only care about the top-k elements
    for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
        dist_T ld = left_dists[i];
        idx_T li = left_indices[i];

        // right array is reverted to allow bitonic comparison
        dist_T rd = right_dists[K - 1 - i];
        idx_T ri = right_indices[K - 1 - i];

        if (rd < ld) {
            dist_out[i] = rd;
            idx_out[i] = ri;
        } else {
            dist_out[i] = ld;
            idx_out[i] = li;
        }
    }

    // Sort Result
    sort_bitonic_sequence_stream<K, dist_T, idx_T>(dist_out, idx_out);
}

} // namespace nnet

#endif
