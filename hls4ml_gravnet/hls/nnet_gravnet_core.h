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
void update_knn(dist_T new_dist, idx_T new_index,
                dist_T knn_dists[CONFIG_T::n_neighbours],
                idx_T knn_indices[CONFIG_T::n_neighbours]) {

    // We keep track of the value currently being "pushed" into the array
    dist_T current_dist = new_dist;
    idx_T current_index = new_index;

    for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
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
template <class coords_T, class feats_T, class output_T, class knn_dist_T, class knn_idx_T, class exp_table_T,
          class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(coords_T coords[CONFIG_T::V * CONFIG_T::S], feats_T feats[CONFIG_T::V * CONFIG_T::F],
                  output_T res[CONFIG_T::V * 2 * CONFIG_T::F]) {
#ifdef __HLS_SYN__
    bool initialized = false;
    exp_table_T exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#endif

    if (!initialized) {
        gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);
        initialized = true;
    }

    knn_dist_T knn_dists[CONFIG_T::V * CONFIG_T::n_neighbours];
    knn_idx_T knn_indices[CONFIG_T::V * CONFIG_T::n_neighbours];

    output_T fmax[CONFIG_T::F];
    output_T fsum[CONFIG_T::F];

    // Initialize arrays
    for (unsigned int v_n = 0; v_n < CONFIG_T::V * CONFIG_T::n_neighbours; v_n++) {
#pragma HLS UNROLL
        knn_dists[v_n] = 32767;
        knn_indices[v_n] = 0;
    }

    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
        for (unsigned int s = 0; s < CONFIG_T::F; s++) {
            fmax[s] = -32768;
            fsum[s] = 0;
        }

        unsigned int row_offset_coords = i * CONFIG_T::S;
        unsigned int knn_offset_i = i * CONFIG_T::n_neighbours;

        for (unsigned int j = 0; j < CONFIG_T::V; j++) {
            if (j > i) {
                unsigned int col_offset_coords = j * CONFIG_T::S;
                unsigned int knn_offset_j = j * CONFIG_T::n_neighbours;

                knn_dist_T dist_sq = 0;

                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
                    knn_dist_T diff = coords[row_offset_coords + s] - coords[col_offset_coords + s];
                    dist_sq += (knn_dist_T)(diff * diff);
                }

                // Pass pointers to the specific sections of the separated arrays
                update_knn<knn_dist_T, knn_idx_T, CONFIG_T>(dist_sq, j, &knn_dists[knn_offset_i], &knn_indices[knn_offset_i]);
                update_knn<knn_dist_T, knn_idx_T, CONFIG_T>(dist_sq, i, &knn_dists[knn_offset_j], &knn_indices[knn_offset_j]);
            }
        }

        for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
            knn_idx_T neighbor_idx = knn_indices[knn_offset_i + n];
            knn_dist_T d = knn_dists[knn_offset_i + n];

            exp_table_idx_T idx = gravnet_idx_from_real_val<knn_dist_T, exp_table_idx_T, CONFIG_T>(d);
            exp_table_T w = exp_table[idx];

            unsigned int neighbor_offset_feats = neighbor_idx * CONFIG_T::F;

            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
                feats_T feat = feats[neighbor_offset_feats + f];
                weighted_feature_T weighted = feat * w;

                fsum[f] += weighted;

                if (weighted > fmax[f]) {
                    fmax[f] = weighted;
                }
            }
        }

        unsigned int out_row_idx = i * (2 * CONFIG_T::F);

        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
            res[out_row_idx + f] = fmax[f];
            res[out_row_idx + CONFIG_T::F + f] = fsum[f] / (output_T)CONFIG_T::n_neighbours;
        }
    }
}

} // namespace nnet

#endif
