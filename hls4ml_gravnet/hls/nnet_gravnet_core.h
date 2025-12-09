#ifndef NNET_GRAVNET_CORE_H_
#define NNET_GRAVNET_CORE_H_

#include "ap_int.h"
#include "nnet_gravnet_bitonic_sort.h"
#include <cmath>
#include <sys/types.h>

namespace nnet {

struct gravnet_core_config {
    static const unsigned V = 128;
    static const unsigned S = 4;
    static const unsigned F = 8;
    static const unsigned n_neighbours = 32;
    static const unsigned exp_table_size = 32;
    static const unsigned exp_table_indexing_shmt = 4;
};

template <class input_T, class exp_table_idx_T, typename CONFIG_T>
inline exp_table_idx_T gravnet_idx_from_real_val(input_T x) {
    if (x < 0)
        x = -x;

    exp_table_idx_T max_idx = CONFIG_T::exp_table_size - 1;

    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);

    if (idx > max_idx)
        return max_idx;

    return (exp_table_idx_T)idx;
}

template <class exp_table_T, typename CONFIG_T>
void gravnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    table_out[0] = 1.0f;
    table_out[CONFIG_T::exp_table_size - 1] = 0.0f;

    for (unsigned i = 1; i < CONFIG_T::exp_table_size - 1; i++) {
#pragma HLS UNROLL
        float val = (float)((ap_fixed<32, 16>)(i + 0.5) >> CONFIG_T::exp_table_indexing_shmt);
        float res = std::exp(-10.0f * val);
        table_out[i] = (exp_table_T)res;
    }
}

template <int N, typename T> T gravnet_sum_tree(T data[N]) {
#pragma HLS INLINE
    T buffer[N / 2];
#pragma HLS ARRAY_PARTITION variable = buffer complete

    for (int i = 0; i < N / 2; i++) {
#pragma HLS UNROLL
        buffer[i] = data[2 * i] + data[2 * i + 1];
    }

    for (int step = 2; step <= (N / 2); step *= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < (N / 2); i += step) {
#pragma HLS UNROLL
            buffer[i] = buffer[i] + buffer[i + (step / 2)];
        }
    }
    return buffer[0];
}

template <int N, typename T> T gravnet_max_tree(T data[N]) {
#pragma HLS INLINE

    T buffer[N / 2];
#pragma HLS ARRAY_PARTITION variable = buffer complete

    for (int i = 0; i < N / 2; i++) {
#pragma HLS UNROLL
        T left = data[2 * i];
        T right = data[2 * i + 1];
        buffer[i] = (left > right) ? left : right;
    }

    for (int step = 2; step <= (N / 2); step *= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < (N / 2); i += step) {
#pragma HLS UNROLL
            T left = buffer[i];
            T right = buffer[i + (step / 2)];
            buffer[i] = (left > right) ? left : right;
        }
    }
    return buffer[0];
}

template <class coords_T, class coords_diff_T, class knn_dist_T, typename CONFIG_T>
void calculate_squared_distances(coords_T coords[CONFIG_T::V * CONFIG_T::S], knn_dist_T squared_dists[CONFIG_T::V],
                                 unsigned int i) {
    for (unsigned int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS UNROLL
        if (i == j) {
            squared_dists[j] = 32000;
            continue;
        }
        knn_dist_T dist_sq = 0;
        for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            coords_diff_T diff = coords[i * CONFIG_T::S + s] - coords[j * CONFIG_T::S + s];
            dist_sq += (knn_dist_T)(diff * diff);
        }
        squared_dists[j] = dist_sq;
    }
}

/**
 * @brief Computes k nearest neighbour pairs (distance, index).
 */
template <class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void select_knn(knn_dist_T squared_distances[CONFIG_T::V], knn_dist_T knn_dists[CONFIG_T::n_neighbours],
                knn_idx_T knn_indices[CONFIG_T::n_neighbours]) {
    const int K = CONFIG_T::n_neighbours;

    // We create NUM_LISTS of size K, since we only need to keep
    // top K values and do not need to sort the entire array.
    const int NUM_LISTS = CONFIG_T::V / K;

    knn_dist_T dist_lists[NUM_LISTS][K];
    knn_idx_T idx_lists[NUM_LISTS][K];
#pragma HLS ARRAY_PARTITION variable = dist_lists complete
#pragma HLS ARRAY_PARTITION variable = idx_lists complete

    // Split the input arrays into lists of length K
    // and sort each list
    for (int i = 0; i < NUM_LISTS; i++) {
#pragma HLS UNROLL
        for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
            dist_lists[i][k] = squared_distances[i * K + k];
            idx_lists[i][k] = (knn_idx_T)(i * K + k);
        }
        bitonic_sort_array<K, knn_dist_T, knn_idx_T>(dist_lists[i], idx_lists[i]);
    }

    // Merge lists by comparing them in a tree like fashion, pushing
    // the smallest values into the list at index 0 (root node)
loop_tree_depth:
    for (int step = 1; step < NUM_LISTS; step *= 2) {
#pragma HLS UNROLL
    loop_tree_width:
        for (int i = 0; i < NUM_LISTS; i += (step * 2)) {
#pragma HLS UNROLL
            int left_i = i;
            int right_i = i + step;
            knn_dist_T tmp_dists[K];
            knn_idx_T tmp_indices[K];
#pragma HLS ARRAY_PARTITION variable = tmp_dists complete
#pragma HLS ARRAY_PARTITION variable = tmp_indices complete

            merge_and_keep_k<K, knn_dist_T, knn_idx_T>(dist_lists[left_i], idx_lists[left_i], dist_lists[right_i],
                                                       idx_lists[right_i], tmp_dists, tmp_indices);
            for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
                dist_lists[left_i][k] = tmp_dists[k];
                idx_lists[left_i][k] = tmp_indices[k];
            }
        }
    }

    // Output
    for (int k = 0; k < K; k++) {
#pragma HLS UNROLL
        knn_dists[k] = dist_lists[0][k];
        knn_indices[k] = idx_lists[0][k];
    }
}

template <class knn_dist_T, class knn_idx_T, class exp_table_idx_T, class exp_table_T, class feats_T,
          class weighted_feature_T, typename CONFIG_T>
void calculate_weighted_features(knn_dist_T knn_dists[CONFIG_T::n_neighbours], knn_idx_T knn_indices[CONFIG_T::n_neighbours],
                                 exp_table_T exp_table[CONFIG_T::exp_table_size], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                                 weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F]) {
loop_weighted_feats_outer:
    for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
        knn_idx_T neighbour_idx = knn_indices[n];
        knn_dist_T d = knn_dists[n];

        unsigned int idx = (unsigned int)gravnet_idx_from_real_val<knn_dist_T, exp_table_idx_T, CONFIG_T>(d);
        exp_table_T w = exp_table[idx];

    loop_weighted_feats_inner:
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            feats_T feat = feats[neighbour_idx * CONFIG_T::F + f];
            weighted_feats[n * CONFIG_T::F + f] = feat * w;
        }
    }
}

template <class weighted_feature_T, class output_T, typename CONFIG_T>
void reduce_features(weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F], output_T fsum[CONFIG_T::F],
                     output_T fmax[CONFIG_T::F]) {
loop_reduce_features:
    for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
        output_T column_for_sum[CONFIG_T::n_neighbours];
        output_T column_for_max[CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = column_for_sum complete
#pragma HLS ARRAY_PARTITION variable = column_for_max complete

        for (int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
            weighted_feature_T val = weighted_feats[n * CONFIG_T::F + f];
            column_for_sum[n] = (output_T)val;
            column_for_max[n] = (output_T)val;
        }

        fsum[f] = gravnet_sum_tree<CONFIG_T::n_neighbours, output_T>(column_for_sum);
        fmax[f] = gravnet_max_tree<CONFIG_T::n_neighbours, output_T>(column_for_max);
    }
}

template <class coords_T, class feats_T, class output_T, class coords_diff_T, class knn_dist_T, class knn_idx_T,
          class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(coords_T coords[CONFIG_T::V * CONFIG_T::S], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                  output_T res[CONFIG_T::V * 2 * CONFIG_T::F]) {
#pragma HLS ARRAY_PARTITION variable = coords complete
#pragma HLS ARRAY_PARTITION variable = feats complete
#pragma HLS ARRAY_PARTITION variable = res complete

#ifdef __HLS_SYN__
    bool initialized = false;
    exp_table_T exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#endif
#pragma HLS ARRAY_PARTITION variable = exp_table complete

    if (!initialized) {
        gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);
        initialized = true;
    }

loop_dist_outer:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE
        unsigned int knn_offset = i * CONFIG_T::n_neighbours;

        // Squared distances
        knn_dist_T current_v_sq_dists[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = current_v_sq_dists complete
        calculate_squared_distances<coords_T, coords_diff_T, knn_dist_T, CONFIG_T>(coords, current_v_sq_dists, i);

        // 2. Select KNN (Parallel Bitonic)
        knn_dist_T knn_dists[CONFIG_T::n_neighbours];
        knn_idx_T knn_indices[CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = knn_dists complete
#pragma HLS ARRAY_PARTITION variable = knn_indices complete
        select_knn<knn_dist_T, knn_idx_T, CONFIG_T>(current_v_sq_dists, knn_dists, knn_indices);

        // 3. Weighted features
        weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = weighted_feats complete
        calculate_weighted_features<knn_dist_T, knn_idx_T, exp_table_idx_T, exp_table_T, feats_T, weighted_feature_T,
                                    CONFIG_T>(knn_dists, knn_indices, exp_table, feats, weighted_feats);

        // 4. Reduce
        output_T fmax[CONFIG_T::F];
        output_T fsum[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = fmax complete
#pragma HLS ARRAY_PARTITION variable = fsum complete
        reduce_features<weighted_feature_T, output_T, CONFIG_T>(weighted_feats, fsum, fmax);

        // 5. Output
        unsigned int out_idx = i * (2 * CONFIG_T::F);
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            res[out_idx + f] = fmax[f];
            res[out_idx + CONFIG_T::F + f] = fsum[f] / (output_T)CONFIG_T::n_neighbours;
        }
    }
}
} // namespace nnet
#endif
