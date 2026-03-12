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

#ifndef NNET_GRAVNET_BITONIC_SORT_STREAM_H_
#define NNET_GRAVNET_BITONIC_SORT_STREAM_H_

namespace nnet {

template <typename dist_T, typename idx_T> void gravnet_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    dist_T temp_d = d1;
    d1 = d2;
    d2 = temp_d;
    idx_T temp_i = i1;
    i1 = i2;
    i2 = temp_i;
}

template <typename dist_T, typename idx_T> void gravnet_compare_and_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    if (d2 < d1) {
        gravnet_swap(d1, i1, d2, i2);
    }
}

template <int N, typename T_Arr_D, typename T_Arr_I> void sort_bitonic_sequence(T_Arr_D &dist, T_Arr_I &idx) {
#pragma HLS INLINE
    for (int stride = N / 2; stride > 0; stride /= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            if ((i % (2 * stride)) < stride) {
                gravnet_compare_and_swap(dist[i], idx[i], dist[i + stride], idx[i + stride]);
            }
        }
    }
}

template <int N, typename T_Arr_D, typename T_Arr_I> void bitonic_sort_array(T_Arr_D &dist, T_Arr_I &idx) {
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
                gravnet_compare_and_swap(dist[idx1], idx[idx1], dist[idx2], idx[idx2]);
            }
            for (int stride = seq_size / 4; stride > 0; stride /= 2) {
#pragma HLS UNROLL
                for (int j = 0; j < seq_size; j++) {
#pragma HLS UNROLL
                    if ((j % (2 * stride)) < stride) {
                        int pos = i + j;
                        gravnet_compare_and_swap(dist[pos], idx[pos], dist[pos + stride], idx[pos + stride]);
                    }
                }
            }
        }
    }
}

template <int K, typename T_Arr_D, typename T_Arr_I>
void merge_and_keep_k(T_Arr_D &left_dists, T_Arr_I &left_indices, T_Arr_D &right_dists, T_Arr_I &right_indices,
                      T_Arr_D &dist_out, T_Arr_I &idx_out) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = dist_out complete
#pragma HLS ARRAY_PARTITION variable = idx_out complete

    for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
        dist_out[i] = (right_dists[K - 1 - i] < left_dists[i]) ? right_dists[K - 1 - i] : left_dists[i];
        idx_out[i] = (right_dists[K - 1 - i] < left_dists[i]) ? right_indices[K - 1 - i] : left_indices[i];
    }
    sort_bitonic_sequence<K>(dist_out, idx_out);
}

} // namespace nnet

#endif
