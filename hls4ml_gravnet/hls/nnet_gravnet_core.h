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

/**
 * @brief Updates the arrays holding the current k-nearest neighbors.
 * * Now separated into two arrays: one for distances, one for indices.
 */
template <class dist_T, class idx_T, typename CONFIG_T>
void update_knn(dist_T new_dist, idx_T new_index, dist_T knn_dists[CONFIG_T::n_neighbours],
                idx_T knn_indices[CONFIG_T::n_neighbours]) {
#pragma HLS INLINE

    // We keep track of the value currently being "pushed" into the array
    dist_T current_dist = new_dist;
    idx_T current_index = new_index;

    for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
        if (current_dist < knn_dists[n]) {
            // Swap Distance
            dist_T tmp_dist = knn_dists[n];
            knn_dists[n] = current_dist;

            // Swap Index (Must happen synchronously with distance)
            idx_T tmp_idx = knn_indices[n];
            knn_indices[n] = current_index;

            // Prepare the old values to be compared against the next neighbor spot
            current_dist = tmp_dist;
            current_index = tmp_idx;
        }
    }
}

/**
 * @brief Implements the core logic of GravNet.
 */
template <class coords_T, class feats_T, class output_T, class coords_diff_T, class knn_dist_T, class knn_idx_T,
          class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(coords_T coords[CONFIG_T::V * CONFIG_T::S], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                  output_T res[CONFIG_T::V * 2 * CONFIG_T::F]) {
#pragma HLS ARRAY_PARTITION variable = coords cyclic factor = CONFIG_T::S dim = 1
#pragma HLS ARRAY_PARTITION variable = feats cyclic factor = CONFIG_T::F dim = 1
#pragma HLS ARRAY_PARTITION variable = res cyclic factor = CONFIG_T::F dim = 1

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

        knn_dist_T local_dists[CONFIG_T::n_neighbours];
        knn_idx_T local_indices[CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = local_dists complete
#pragma HLS ARRAY_PARTITION variable = local_indices complete

    loop_init_knn_dist_idx:
        for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
            local_dists[n] = 32767;
            local_indices[n] = 0;
        }

    loop_dist_inner:
        for (unsigned int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS PIPELINE II = 1
            if (i == j)
                continue; // no self-comparison
            knn_dist_T dist_sq = 0;

        loop_dist_sq:
            for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                coords_diff_T diff = coords[i * CONFIG_T::S + s] - coords[j * CONFIG_T::S + s];
                dist_sq += (knn_dist_T)(diff * diff);
            }

            update_knn<knn_dist_T, knn_idx_T, CONFIG_T>(dist_sq, j, local_dists, local_indices);
        }

        output_T fmax[CONFIG_T::F];
        output_T fsum[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = fmax complete
#pragma HLS ARRAY_PARTITION variable = fsum complete

        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            fmax[f] = -32768;
            fsum[f] = 0;
        }

    loop_agg_neighbors:
        for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
            knn_idx_T neighbour_idx = local_indices[n];
            knn_dist_T d = local_dists[n];

            unsigned int idx = gravnet_idx_from_real_val<knn_dist_T, exp_table_idx_T, CONFIG_T>(d);
            exp_table_T w = exp_table[idx];

        loop_agg_feats:
            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
                feats_T feat = feats[neighbour_idx * CONFIG_T::F + f];
                weighted_feature_T weighted = feat * w;

                fsum[f] += weighted;
                if (weighted > fmax[f])
                    fmax[f] = weighted;
            }
        }

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
