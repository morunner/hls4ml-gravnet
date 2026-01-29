#ifndef NNET_GRAVNET_CORE_STREAM_H_
#define NNET_GRAVNET_CORE_STREAM_H_

#include "hls_stream.h"
#include "nnet_gravnet_core.h"

namespace nnet {

template <class coords_T, class feats_T, class coord_val_T, class feat_val_T, typename CONFIG_T>
void read_inputs(hls::stream<coords_T> &coords_stream, hls::stream<feats_T> &feats_stream,
                 coord_val_T coords_buffer[CONFIG_T::V * CONFIG_T::S], feat_val_T feats_buffer[CONFIG_T::V * CONFIG_T::F]) {
#pragma HLS INLINE off

ReadLoop:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1

        coords_T c_pack = coords_stream.read();
        feats_T f_pack = feats_stream.read();

        for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            coords_buffer[i * CONFIG_T::S + s] = c_pack[s];
        }
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            feats_buffer[i * CONFIG_T::F + f] = f_pack[f];
        }
    }
}

template <class output_T, class coord_val_T, class feat_val_T, class accum_T, class coords_diff_T, class knn_dist_T,
          class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void compute_outputs(coord_val_T coords_buffer[CONFIG_T::V * CONFIG_T::S],
                     feat_val_T feats_buffer[CONFIG_T::V * CONFIG_T::F], exp_table_T exp_table[CONFIG_T::exp_table_size],
                     hls::stream<output_T> &res_stream) {
#pragma HLS INLINE off

    typedef typename output_T::value_type out_val_t;

ComputeLoop:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1

        knn_dist_T current_v_sq_dists[CONFIG_T::V];
#pragma HLS ARRAY_PARTITION variable = current_v_sq_dists complete
        calculate_squared_distances<coord_val_T, coords_diff_T, knn_dist_T, CONFIG_T>(coords_buffer, current_v_sq_dists, i);

        knn_dist_T knn_dists[CONFIG_T::n_neighbours];
        knn_idx_T knn_indices[CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = knn_dists complete
#pragma HLS ARRAY_PARTITION variable = knn_indices complete
        select_knn<knn_dist_T, knn_idx_T, CONFIG_T>(current_v_sq_dists, knn_dists, knn_indices);

        out_val_t fmax[CONFIG_T::F];
        accum_T fsum[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = fmax complete
#pragma HLS ARRAY_PARTITION variable = fsum complete
        apply_weights_and_reduce<knn_dist_T, knn_idx_T, exp_table_idx_T, exp_table_T, feat_val_T, weighted_feature_T,
                                 accum_T, out_val_t, CONFIG_T>(knn_dists, knn_indices, exp_table, feats_buffer, fsum, fmax);

        output_T res_pack;
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            res_pack[f] = fmax[f];
            res_pack[CONFIG_T::F + f] = fsum[f] / (out_val_t)CONFIG_T::n_neighbours;
        }
        res_stream.write(res_pack);
    }
}

template <class coords_T, class feats_T, class output_T, class accum_T, class coords_diff_T, class knn_dist_T,
          class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(hls::stream<coords_T> &coords_stream, hls::stream<feats_T> &feats_stream,
                  hls::stream<output_T> &res_stream) {
#pragma HLS DATAFLOW

    typedef typename coords_T::value_type coord_val_t;
    typedef typename feats_T::value_type feat_val_t;

    coord_val_t coords_buffer[CONFIG_T::V * CONFIG_T::S];
    feat_val_t feats_buffer[CONFIG_T::V * CONFIG_T::F];

#pragma HLS ARRAY_PARTITION variable = coords_buffer complete
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete

    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);

    read_inputs<coords_T, feats_T, coord_val_t, feat_val_t, CONFIG_T>(coords_stream, feats_stream, coords_buffer,
                                                                      feats_buffer);

    compute_outputs<output_T, coord_val_t, feat_val_t, accum_T, coords_diff_T, knn_dist_T, knn_idx_T, exp_table_T,
                    exp_table_idx_T, weighted_feature_T, CONFIG_T>(coords_buffer, feats_buffer, exp_table, res_stream);
}

} // namespace nnet
#endif
