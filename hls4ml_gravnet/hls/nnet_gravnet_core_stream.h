#ifndef NNET_GRAVNET_CORE_STREAM_H_
#define NNET_GRAVNET_CORE_STREAM_H_

#include "hls_stream.h"
#include "nnet_gravnet_bitonic_sort_stream.h"
#include "nnet_gravnet_core_utils.h"
#include "nnet_types.h"

namespace nnet {

template <typename CONFIG_T, class coords_T, class feats_T>
void buffer_inputs(hls::stream<coords_T> &coords_in, hls::stream<feats_T> &feats_in, hls::stream<coords_T> &coords_out,
                   hls::stream<feats_T> &feats_out) {
#pragma HLS INLINE off
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1
        coords_out.write(coords_in.read());
        feats_out.write(feats_in.read());
    }
}

template <class coords_T, class feats_T, class coord_val_T, class feat_val_T, typename CONFIG_T>
void read_inputs(hls::stream<coords_T> &coords_stream, hls::stream<feats_T> &feats_stream,
                 coord_val_T coords_buffer[CONFIG_T::V][CONFIG_T::S],
                 feat_val_T feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F]) {
#pragma HLS INLINE off

ReadLoop:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1

        coords_T coord = coords_stream.read();
        feats_T feat = feats_stream.read();

        for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            coords_buffer[i][s] = coord[s];
        }

        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            feat_val_T val = feat[f];

            for (unsigned int n_read = 0; n_read < CONFIG_T::n_neighbours; n_read++) {
#pragma HLS UNROLL
                feats_buffer[n_read][i][f] = val;
            }
        }
    }
}

template <class coords_T, class coords_diff_T, class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void calculate_distances(
    typename coords_T::value_type coords_buffer[CONFIG_T::V][CONFIG_T::S],
    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[CONFIG_T::V / CONFIG_T::n_neighbours]) {
#pragma HLS ARRAY_PARTITION variable = coords_buffer dim = 0 complete

    constexpr unsigned int num_leaves = CONFIG_T::V / CONFIG_T::n_neighbours;

    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1

        nnet::array<knn_dist_T, CONFIG_T::n_neighbours> leaf_dists[num_leaves];
        nnet::array<knn_idx_T, CONFIG_T::n_neighbours> leaf_idxs[num_leaves];
#pragma HLS ARRAY_PARTITION variable = leaf_dists complete
#pragma HLS ARRAY_PARTITION variable = leaf_idxs complete

        for (unsigned int l = 0; l < num_leaves; l++) {
#pragma HLS UNROLL
            for (unsigned int k = 0; k < CONFIG_T::n_neighbours; k++) {
#pragma HLS UNROLL
                unsigned int j = l * CONFIG_T::n_neighbours + k;

                knn_dist_T dist = 0;
                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                    dist += CONFIG_T::template distance_fn<typename coords_T::value_type, knn_dist_T, coords_diff_T>::dist(
                        coords_buffer[i][s], coords_buffer[j][s]);
                }

                leaf_dists[l][k] = (i == j) ? gravnet_core_limits<knn_dist_T>::max_val() : dist;
                leaf_idxs[l][k] = (knn_idx_T)j;
            }
        }
        for (unsigned int L = 0; L < num_leaves; L++) {
#pragma HLS UNROLL
            dist_streams[L].write(leaf_dists[L]);
            idx_streams[L].write(leaf_idxs[L]);
        }
    }
}

template <class feat_val_T, class output_T, class accum_T, class knn_dist_T, class knn_idx_T, class exp_table_T,
          class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void apply_weights_and_reduce(hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> &knn_dists,
                              hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> &knn_indices,
                              feat_val_T feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F],
                              exp_table_T exp_table[CONFIG_T::exp_table_size], hls::stream<output_T> &out_stream) {
    accum_T acc_sum[CONFIG_T::F];
    accum_T acc_max[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = acc_sum complete
#pragma HLS ARRAY_PARTITION variable = acc_max complete

ReduceVertexLoop:
    for (unsigned int i = 0; i < CONFIG_T::V; i++) {
#pragma HLS PIPELINE II = 1

        output_T out;

        for (int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            acc_sum[f] = 0;
            acc_max[f] = gravnet_core_limits<accum_T>::min_val();
        }

        nnet::array<knn_dist_T, CONFIG_T::n_neighbours> dists = knn_dists.read();
        nnet::array<knn_idx_T, CONFIG_T::n_neighbours> indices = knn_indices.read();

        for (unsigned int n = 0; n < CONFIG_T::n_neighbours; n++) {
#pragma HLS UNROLL
            knn_dist_T d = dists[n];
            unsigned int idx = indices[n];

            unsigned int table_idx = gravnet_idx_from_real_val<knn_dist_T, exp_table_idx_T, CONFIG_T>(d);
            exp_table_T w = exp_table[table_idx];

            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
                feat_val_T feat = feats_buffer[n][idx][f];
                weighted_feature_T val = (weighted_feature_T)(feat * w);

                acc_sum[f] += val;
                if (val > acc_max[f]) {
                    acc_max[f] = val;
                }
            }
        }

        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            out[f] = acc_max[f];
            out[CONFIG_T::F + f] = acc_sum[f] / (accum_T)CONFIG_T::n_neighbours;
        }
        out_stream.write(out);
    }
}

template <class coords_T, class feats_T, class output_T, class accum_T, class coords_diff_T, class knn_dist_T,
          class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(hls::stream<coords_T> &coords_stream, hls::stream<feats_T> &feats_stream,
                  hls::stream<output_T> &out_stream) {
#pragma HLS DATAFLOW

    constexpr unsigned num_leaves = CONFIG_T::V / CONFIG_T::n_neighbours;

    typedef typename coords_T::value_type coord_val_t;
    typedef typename feats_T::value_type feat_val_t;

    hls::stream<coords_T> coords_stream_buffer;
    hls::stream<feats_T> feats_stream_buffer;
#pragma HLS STREAM variable = coords_stream_buffer depth = 2
#pragma HLS STREAM variable = feats_stream_buffer depth = 2

    coord_val_t coords_buffer[CONFIG_T::V][CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = coords_buffer complete dim = 0

    // Explicitly replicate this buffer such that it can be read for each neighbour iteration
    // in parallel. This is also explicitly put into BRAM for efficient routing.
    feat_val_t feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F];
#pragma HLS BIND_STORAGE variable = feats_buffer type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete dim = 1
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete dim = 3

    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[num_leaves];
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[num_leaves];
#pragma HLS STREAM variable = dist_streams depth = 16
#pragma HLS STREAM variable = idx_streams depth = 16

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> knn_dists;
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> knn_indices;
#pragma HLS STREAM variable = knn_dists depth = 16
#pragma HLS STREAM variable = knn_indices depth = 16

    // Buffering the inputs at the boundary of this function helps timing closure
    buffer_inputs<CONFIG_T, coords_T, feats_T>(coords_stream, feats_stream, coords_stream_buffer, feats_stream_buffer);

    read_inputs<coords_T, feats_T, coord_val_t, feat_val_t, CONFIG_T>(coords_stream_buffer, feats_stream_buffer,
                                                                      coords_buffer, feats_buffer);

    calculate_distances<coords_T, coords_diff_T, knn_dist_T, knn_idx_T, CONFIG_T>(coords_buffer, dist_streams, idx_streams);

    select_knn_tree<knn_dist_T, knn_idx_T, CONFIG_T>(dist_streams, idx_streams, knn_dists, knn_indices);

    apply_weights_and_reduce<feat_val_t, output_T, accum_T, knn_dist_T, knn_idx_T, exp_table_T, exp_table_idx_T,
                             weighted_feature_T, CONFIG_T>(knn_dists, knn_indices, feats_buffer, exp_table, out_stream);
}

} // namespace nnet
#endif
