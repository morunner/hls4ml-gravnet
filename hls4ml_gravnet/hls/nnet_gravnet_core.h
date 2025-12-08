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

// GravNet Core Logic

template <class coords_T, class coords_diff_T, class knn_dist_T, typename CONFIG_T>
void calculate_squared_distances(coords_T coords[CONFIG_T::V * CONFIG_T::S], knn_dist_T squared_dists[CONFIG_T::V],
                                 unsigned int i) {
loop_dist_sq:
    for (unsigned int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS UNROLL
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

template <typename dist_T, typename idx_T> void compare_and_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    if (d2 < d1) {
        // Swap Distances
        dist_T temp_d = d1;
        d1 = d2;
        d2 = temp_d;

        // Swap Indices
        idx_T temp_i = i1;
        i1 = i2;
        i2 = temp_i;
    }
}

// Bitonic sort of 4 elements
template <typename dist_T, typename idx_T> void bitonic_sort_4(dist_T d[4], idx_T i[4]) {
#pragma HLS INLINE
    // Create bitonic sequence of length 4
    compare_and_swap(d[0], i[0], d[1], i[1]); // Left side (ascending)
    compare_and_swap(d[3], i[3], d[2], i[2]); // Right side (descending)

    // Merge bitonic sequences (sort)
    // Stride 2
    compare_and_swap(d[0], i[0], d[2], i[2]);
    compare_and_swap(d[1], i[1], d[3], i[3]);
    // Stride 1
    compare_and_swap(d[0], i[0], d[1], i[1]);
    compare_and_swap(d[2], i[2], d[3], i[3]);
}

template <typename dist_T, typename idx_T>
void bitonic_merge(dist_T left_dist[4], idx_T left_idx[4], dist_T right_dist[4], idx_T right_idx[4], dist_T dist_res[4],
                   idx_T idx_res[4]) {
#pragma HLS INLINE
    dist_T tmp_dist[4];
    idx_T tmp_idx[4];
#pragma HLS ARRAY_PARTITION variable = tmp_dist complete
#pragma HLS ARRAY_PARTITION variable = tmp_idx complete

    // Half-cleaner
    //  This merges two bitonic sequences by comparing the corresponding values.
    //  tmp_dist, tmp_idx are assigned the smaller values and form again a bitonic
    //  sequence. Since we are only interested in the smallest values, we discard
    //  the right side.
    for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
        dist_T left_d = left_dist[k];
        idx_T left_i = left_idx[k];

        // Right side is in descending order
        // to ensure bitonic sequence property (left and right are both sorted)
        dist_T right_d = right_dist[3 - k];
        idx_T right_i = right_idx[3 - k];

        // Merge and discard larger (right) of the two
        // bitonic sequences
        if (right_d < left_d) {
            tmp_dist[k] = right_d;
            tmp_idx[k] = right_i;
        } else {
            tmp_dist[k] = left_d;
            tmp_idx[k] = left_i;
        }
    }

    // Now we need to sort the new bitonic sequence by comparing the corresponding elements
    // of ascending and descending side of the sequence.
    compare_and_swap(tmp_dist[0], tmp_idx[0], tmp_dist[2], tmp_idx[2]);
    compare_and_swap(tmp_dist[1], tmp_idx[1], tmp_dist[3], tmp_idx[3]);
    compare_and_swap(tmp_dist[0], tmp_idx[0], tmp_dist[1], tmp_idx[1]);
    compare_and_swap(tmp_dist[2], tmp_idx[2], tmp_dist[3], tmp_idx[3]);

    // Write result to Output
    for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
        dist_res[k] = tmp_dist[k];
        idx_res[k] = tmp_idx[k];
    }
}

template <class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void select_knn(knn_dist_T squared_distances[CONFIG_T::V], knn_dist_T knn_dists[CONFIG_T::n_neighbours],
                knn_idx_T knn_indices[CONFIG_T::n_neighbours]) {

    // Number of buffers for bitonic sort
    // We use 4 elements per list
    const int NUM_LISTS = CONFIG_T::V / 4;

    knn_dist_T list_dist[NUM_LISTS][4];
    knn_idx_T list_idx[NUM_LISTS][4];
#pragma HLS ARRAY_PARTITION variable = list_dist complete
#pragma HLS ARRAY_PARTITION variable = list_idx complete

    // Create NUM_LISTS sorted lists of length 4
    for (int i = 0; i < NUM_LISTS; i++) {
#pragma HLS UNROLL
        for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
            list_dist[i][k] = squared_distances[i * 4 + k];
            list_idx[i][k] = (knn_idx_T)(i * 4 + k);
        }
        bitonic_sort_4(list_dist[i], list_idx[i]);
    }

    // Tree reduction
    //  Now that we have NUM_LISTS sorted lists, we can merge them in a binary tree manner
    for (int step = 1; step < NUM_LISTS; step *= 2) {
#pragma HLS UNROLL

        // Merge sorted lists in a binary tree-like manner
        for (int i = 0; i < NUM_LISTS; i += (step * 2)) {
#pragma HLS UNROLL

            // Select consecutive 2 lists
            int left_idx = i;
            int right_idx = i + step;

            knn_dist_T tmp_dist[4];
            knn_idx_T tmp_idx[4];
#pragma HLS ARRAY_PARTITION variable = tmp_dist complete
#pragma HLS ARRAY_PARTITION variable = tmp_idx complete

            // Each list contains 4 elements -> 8 elements per bitonic_merge pass. We keep the smallest 4 elements per iteration.
            bitonic_merge(list_dist[left_idx], list_idx[left_idx], list_dist[right_idx], list_idx[right_idx], tmp_dist,
                          tmp_idx);

            // Only keep the left half of the bitonic sequence,
            // since we care about the smallest values (= nearest neighbours)
            for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
                list_dist[left_idx][k] = tmp_dist[k];
                list_idx[left_idx][k] = tmp_idx[k];
            }
        }
    }

    // Even though the whole
    for (int k = 0; k < CONFIG_T::n_neighbours; k++) {
#pragma HLS UNROLL
        knn_dists[k] = list_dist[0][k];
        knn_indices[k] = list_idx[0][k];
    }
}

template <class knn_dist_T, class knn_idx_T, class exp_table_idx_T, class exp_table_T, class feats_T,
          class weighted_feature_T, typename CONFIG_T>
void calculate_weighted_features(knn_dist_T knn_dists[CONFIG_T::n_neighbours], knn_idx_T knn_indices[CONFIG_T::n_neighbours],
                                 exp_table_T exp_table[CONFIG_T::exp_table_size], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                                 weighted_feature_T weighted_feats[CONFIG_T::n_neighbours * CONFIG_T::F]) {
loop_weighted_feats_outer:
    for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
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
#pragma HLS PIPELINE
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
