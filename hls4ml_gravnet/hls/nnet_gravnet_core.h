#ifndef NNET_GRAVNET_CORE_H_
#define NNET_GRAVNET_CORE_H_

#include "ap_int.h"
#include <cmath>
#include <sys/types.h>
namespace nnet {

struct gravnet_core_config {
    static const unsigned V = 128;
    static const unsigned S = 4;
    static const unsigned F = 8;
    static const unsigned n_neighbours = 4;
    static const unsigned exp_table_size = 32;
    static const unsigned exp_table_indexing_shmt = 4;
};

// Utils

/**
 * @brief Converts a real value to an index for the exponent lookup table.
 */
template <class input_T, class exp_table_idx_T, typename CONFIG_T>
inline exp_table_idx_T gravnet_idx_from_real_val(input_T x) {
    if (x < 0)
        x = -x;

    exp_table_idx_T max_idx = CONFIG_T::exp_table_size - 1;

    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);

    if (idx > max_idx) {
        return max_idx;
    }
    return (exp_table_idx_T)idx;
}

/**
 * @brief Initializes the exponent lookup table.
 */
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

    T buffer[N];
#pragma HLS ARRAY_PARTITION variable = buffer complete

    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
        buffer[i] = data[i];
    }

    for (int step = 2; step <= N; step *= 2) {
#pragma HLS UNROLL

        for (int i = 0; i < N; i += step) {
#pragma HLS UNROLL
            buffer[i] = buffer[i] + buffer[i + (step / 2)];
        }
    }

    return buffer[0];
}

template <int N, typename T> T gravnet_max_tree(T data[N]) {
#pragma HLS INLINE

    T buffer[N];
#pragma HLS ARRAY_PARTITION variable = buffer complete

    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
        buffer[i] = data[i];
    }

    for (int step = 2; step <= N; step *= 2) {
#pragma HLS UNROLL

        for (int i = 0; i < N; i += step) {
#pragma HLS UNROLL

            T left = buffer[i];
            T right = buffer[i + (step / 2)];

            buffer[i] = (left > right) ? left : right;
        }
    }

    // The root (maximum value) ends up at index 0
    return buffer[0];
}

template <int N, typename dist_T, typename idx_T>
void gravnet_min_tree(dist_T input_dists[N], dist_T &out_min_dist, idx_T &out_min_idx) {
#pragma HLS INLINE

    dist_T tree_dist[N];
    idx_T tree_idx[N];
#pragma HLS ARRAY_PARTITION variable = tree_dist complete
#pragma HLS ARRAY_PARTITION variable = tree_idx complete

    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
        tree_dist[i] = input_dists[i];
        tree_idx[i] = i;
    }

    for (int step = 2; step <= N; step *= 2) {
#pragma HLS UNROLL

        for (int i = 0; i < N; i += step) {
#pragma HLS UNROLL

            int left = i;
            int right = i + (step / 2);

            if (tree_dist[right] < tree_dist[left]) {
                tree_dist[left] = tree_dist[right];
                tree_idx[left] = tree_idx[right];
            }
        }
    }

    // Root is at index 0
    out_min_dist = tree_dist[0];
    out_min_idx = tree_idx[0];
}

// GravNet Core Logic
template <class coords_T, class coords_diff_T, class knn_dist_T, typename CONFIG_T>
void calculate_squared_distances(coords_T coords[CONFIG_T::V * CONFIG_T::S], knn_dist_T squared_dists[CONFIG_T::V],
                                 unsigned int i) {
loop_dist_sq:
    for (unsigned int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS PIPELINE II = 1
        if (i == j) {
            squared_dists[j] = 32767;
            continue; // no self-comparison
        }

        knn_dist_T dist_sq = 0;

    loop_dist_sq_accum:
        for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            coords_diff_T diff = coords[i * CONFIG_T::S + s] - coords[j * CONFIG_T::S + s];
            dist_sq += (knn_dist_T)(diff * diff);
        }

        squared_dists[j] = dist_sq;
    }
}

template <class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void select_knn(knn_dist_T squared_distances[CONFIG_T::V], knn_dist_T knn_dists[CONFIG_T::n_neighbours],
                knn_idx_T knn_indices[CONFIG_T::n_neighbours]) {
    bool is_selected[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = is_selected complete

init_mask:
    for (int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS UNROLL
        is_selected[i] = false;
    }
loop_select_top_k:
    for (unsigned int k = 0; k < CONFIG_T::n_neighbours; k++) {
#pragma HLS PIPELINE II = 1
        knn_dist_T masked_dists[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = masked_dists complete

    loop_mask_inputs:
        for (int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS UNROLL
            masked_dists[j] = is_selected[j] ? (knn_dist_T)32000 : squared_distances[j];
        }

        knn_dist_T min_dist;
        knn_idx_T min_idx;

        gravnet_min_tree<CONFIG_T::V, knn_dist_T, knn_idx_T>(masked_dists, min_dist, min_idx);

        knn_dists[k] = min_dist;
        knn_indices[k] = min_idx;

        if (min_idx >= 0 && min_idx < CONFIG_T::V) {
            is_selected[min_idx] = true;
        }
    }
}

template <class knn_dist_T, class knn_idx_T, class exp_table_idx_T, class exp_table_T, class feats_T,
          class weighted_feature_T, typename CONFIG_T>
void calculate_weighted_features(knn_dist_T knn_dists[CONFIG_T::n_neighbours], knn_idx_T knn_indices[CONFIG_T::n_neighbours],
                                 exp_table_T exp_table[CONFIG_T::exp_table_size], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                                 weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F]) {
loop_weighted_feats_outer:
    for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS PIPELINE II = 1
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
#pragma HLS PIPELINE II = 1
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

/**
 * @brief Implements the core logic of GravNet.
 */
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

// Initialize arrays
loop_dist_outer:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
        unsigned int knn_offset = i * CONFIG_T::n_neighbours;

        // Squared distances
        knn_dist_T current_v_sq_dists[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = current_v_sq_dists complete
        calculate_squared_distances<coords_T, coords_diff_T, knn_dist_T, CONFIG_T>(coords, current_v_sq_dists, i);

        // K-neares neighbours
        knn_dist_T knn_dists[CONFIG_T::n_neighbours];
        knn_idx_T knn_indices[CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = knn_dists complete
#pragma HLS ARRAY_PARTITION variable = knn_indices complete
        select_knn<knn_dist_T, knn_idx_T, CONFIG_T>(current_v_sq_dists, knn_dists, knn_indices);

        // Weighted features
        weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = weighted_feats complete
        calculate_weighted_features<knn_dist_T, knn_idx_T, exp_table_idx_T, exp_table_T, feats_T, weighted_feature_T,
                                    CONFIG_T>(knn_dists, knn_indices, exp_table, feats, weighted_feats);

        // Calculate output features
        output_T fmax[CONFIG_T::F];
        output_T fsum[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = fmax complete
#pragma HLS ARRAY_PARTITION variable = fsum complete
        reduce_features<weighted_feature_T, output_T, CONFIG_T>(weighted_feats, fsum, fmax);

        // Assign output features to output
        unsigned int out_idx = i * (2 * CONFIG_T::F);
    loop_res:
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            res[out_idx + f] = fmax[f];
            res[out_idx + CONFIG_T::F + f] = fsum[f] / (output_T)CONFIG_T::n_neighbours;
        }
    }
}

} // namespace nnet

#endif
