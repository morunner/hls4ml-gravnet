#ifndef NNET_GRAVNET_CORE_STREAM_H_
#define NNET_GRAVNET_CORE_STREAM_H_

#include "hls_stream.h"
#include "nnet_gravnet_bitonic_sort_stream.h"
#include "nnet_gravnet_core_utils.h"
#include "nnet_types.h"

namespace nnet {

template <unsigned P, class coords_pack_T, class feats_pack_T, typename CONFIG_T>
void fork_inputs(hls::stream<coords_pack_T> &coords_in, hls::stream<feats_pack_T> &feats_in,
                 hls::stream<coords_pack_T> coords_out[P], hls::stream<feats_pack_T> feats_out[P]) {
    for (unsigned int i = 0; i < CONFIG_T::V / P; i++) {
#pragma HLS PIPELINE II = 1

        coords_pack_T c_pack = coords_in.read();
        feats_pack_T f_pack = feats_in.read();

        for (unsigned int p = 0; p < P; p++) {
#pragma HLS UNROLL
            coords_out[p].write(c_pack);
            feats_out[p].write(f_pack);
        }
    }
}

template <unsigned P, class output_single_T, class output_pack_T, typename CONFIG_T>
void join_outputs(hls::stream<output_single_T> res_in[P], hls::stream<output_pack_T> &res_out) {
    for (unsigned int i = 0; i < CONFIG_T::V / P; i++) {
#pragma HLS PIPELINE II = 1

        output_pack_T out_pack;

        for (unsigned int p = 0; p < P; p++) {
#pragma HLS UNROLL
            output_single_T single_res = res_in[p].read();

            for (unsigned int f = 0; f < 2 * CONFIG_T::F; f++) {
#pragma HLS UNROLL
                out_pack[p * (2 * CONFIG_T::F) + f] = single_res[f];
            }
        }
        res_out.write(out_pack);
    }
}

template <unsigned P, class coords_pack_T, class feats_pack_T, class coord_val_T, class feat_val_T, typename CONFIG_T>
void read_inputs(hls::stream<coords_pack_T> &coords_stream, hls::stream<feats_pack_T> &feats_stream,
                 coord_val_T coords_buffer[CONFIG_T::V][CONFIG_T::S],
                 feat_val_T feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F]) {
#pragma HLS INLINE off

ReadLoop:
    for (unsigned int i = 0; i < CONFIG_T::V / P; i++) {
#pragma HLS PIPELINE II = 1

        coords_pack_T c_pack = coords_stream.read();
        feats_pack_T f_pack = feats_stream.read();

        for (unsigned int p = 0; p < P; p++) {
#pragma HLS UNROLL
            unsigned int v_idx = i * P + p;

            for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                coords_buffer[v_idx][s] = c_pack[p * CONFIG_T::S + s];
            }

            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
                feat_val_T val = f_pack[p * CONFIG_T::F + f];

                for (unsigned int n_read = 0; n_read < CONFIG_T::n_neighbours; n_read++) {
#pragma HLS UNROLL
                    feats_buffer[n_read][v_idx][f] = val;
                }
            }
        }
    }
}

template <unsigned P, class coords_pack_T, class coords_diff_T, class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void calculate_distances(
    typename coords_pack_T::value_type coords_buffer[CONFIG_T::V][CONFIG_T::S],
    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    unsigned int p_idx) {

#pragma HLS ARRAY_PARTITION variable = coords_buffer dim = 0 complete
    constexpr unsigned int num_chunks = CONFIG_T::V / CONFIG_T::n_neighbours;

VertexLoop:
    for (unsigned int i_local = 0; i_local < CONFIG_T::V / P; i_local++) {
#pragma HLS PIPELINE II = 1
        unsigned int i = (i_local * P) + p_idx;

        nnet::array<knn_dist_T, CONFIG_T::n_neighbours> chunks_dist[num_chunks];
        nnet::array<knn_idx_T, CONFIG_T::n_neighbours> chunks_idx[num_chunks];
#pragma HLS ARRAY_PARTITION variable = chunks_dist complete
#pragma HLS ARRAY_PARTITION variable = chunks_idx complete

    TargetLoop_L:
        for (unsigned int L = 0; L < num_chunks; L++) {
#pragma HLS UNROLL
        TargetLoop_k:
            for (unsigned int k = 0; k < CONFIG_T::n_neighbours; k++) {
#pragma HLS UNROLL
                unsigned int j = L * CONFIG_T::n_neighbours + k;

                knn_dist_T dist = 0;
                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                    dist +=
                        CONFIG_T::template distance_fn<typename coords_pack_T::value_type, knn_dist_T, coords_diff_T>::dist(
                            coords_buffer[i][s], coords_buffer[j][s]);
                }

                chunks_dist[L][k] = (i == j) ? gravnet_core_limits<knn_dist_T>::max_val() : dist;
                chunks_idx[L][k] = (knn_idx_T)j;
            }
        }
        for (unsigned int L = 0; L < num_chunks; L++) {
#pragma HLS UNROLL
            dist_streams[L].write(chunks_dist[L]);
            idx_streams[L].write(chunks_idx[L]);
        }
    }
}

template <unsigned P, class feat_val_T, class output_single_T, class accum_T, class knn_dist_T, class knn_idx_T,
          class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void apply_weights_and_reduce(hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> &knn_dists,
                              hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> &knn_indices,
                              feat_val_T feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F],
                              exp_table_T exp_table[CONFIG_T::exp_table_size], hls::stream<output_single_T> &res_stream) {
    accum_T acc_sum[CONFIG_T::F];
    accum_T acc_max[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = acc_sum complete
#pragma HLS ARRAY_PARTITION variable = acc_max complete

ReduceVertexLoop:
    for (unsigned int i_local = 0; i_local < CONFIG_T::V / P; i_local++) {
#pragma HLS PIPELINE II = 1
        output_single_T out_single;

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
            out_single[f] = acc_max[f];
            out_single[CONFIG_T::F + f] = acc_sum[f] / (accum_T)CONFIG_T::n_neighbours;
        }
        res_stream.write(out_single);
    }
}

template <unsigned P, class coords_pack_T, class feats_pack_T, class output_single_T, class accum_T, class coords_diff_T,
          class knn_dist_T, class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T,
          typename CONFIG_T>
void gravnet_core_worker(hls::stream<coords_pack_T> &coords_stream, hls::stream<feats_pack_T> &feats_stream,
                         hls::stream<output_single_T> &res_stream, unsigned int p_idx) {
#pragma HLS DATAFLOW

    constexpr unsigned num_leaves = CONFIG_T::V / CONFIG_T::n_neighbours;

    typedef typename coords_pack_T::value_type coord_val_t;
    typedef typename feats_pack_T::value_type feat_val_t;

    coord_val_t coords_buffer[CONFIG_T::V][CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = coords_buffer complete dim = 0

    feat_val_t feats_buffer[CONFIG_T::n_neighbours][CONFIG_T::V][CONFIG_T::F];
#pragma HLS BIND_STORAGE variable = feats_buffer type = ram_2p impl = bram
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete dim = 1
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete dim = 3

    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[num_leaves];
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[num_leaves];

#pragma HLS STREAM variable = dist_streams depth = 32
#pragma HLS STREAM variable = idx_streams depth = 32

#pragma HLS BIND_STORAGE variable = dist_streams type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = idx_streams type = fifo impl = srl

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> knn_dists;
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> knn_indices;

#pragma HLS STREAM variable = knn_dists depth = 32
#pragma HLS STREAM variable = knn_indices depth = 32

#pragma HLS BIND_STORAGE variable = knn_dists type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = knn_indices type = fifo impl = srl

    read_inputs<P, coords_pack_T, feats_pack_T, coord_val_t, feat_val_t, CONFIG_T>(coords_stream, feats_stream,
                                                                                   coords_buffer, feats_buffer);

    calculate_distances<P, coords_pack_T, coords_diff_T, knn_dist_T, knn_idx_T, CONFIG_T>(coords_buffer, dist_streams,
                                                                                          idx_streams, p_idx);

    select_knn_tree<P, knn_dist_T, knn_idx_T, CONFIG_T>(dist_streams, idx_streams, knn_dists, knn_indices);

    apply_weights_and_reduce<P, feat_val_t, output_single_T, accum_T, knn_dist_T, knn_idx_T, exp_table_T, exp_table_idx_T,
                             weighted_feature_T, CONFIG_T>(knn_dists, knn_indices, feats_buffer, exp_table, res_stream);
}

template <class coords_pack_T, class feats_pack_T, class output_pack_T, class accum_T, class coords_diff_T, class knn_dist_T,
          class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(hls::stream<coords_pack_T> &coords_stream, hls::stream<feats_pack_T> &feats_stream,
                  hls::stream<output_pack_T> &res_stream) {
#pragma HLS DATAFLOW

    constexpr unsigned int TOTAL_COORDS = coords_pack_T::size;
    constexpr unsigned int P = TOTAL_COORDS / CONFIG_T::S;

    typedef nnet::array<typename output_pack_T::value_type, 2 * CONFIG_T::F> output_single_T;

    hls::stream<coords_pack_T> coords_forked[P];
    hls::stream<feats_pack_T> feats_forked[P];
    hls::stream<output_single_T> res_forked[P];

#pragma HLS STREAM variable = coords_forked depth = 32
#pragma HLS STREAM variable = feats_forked depth = 32
#pragma HLS STREAM variable = res_forked depth = 32
#pragma HLS BIND_STORAGE variable = coords_forked type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = feats_forked type = fifo impl = srl
#pragma HLS BIND_STORAGE variable = res_forked type = fifo impl = srl

    fork_inputs<P, coords_pack_T, feats_pack_T, CONFIG_T>(coords_stream, feats_stream, coords_forked, feats_forked);

    for (unsigned int p = 0; p < P; p++) {
#pragma HLS UNROLL
        gravnet_core_worker<P, coords_pack_T, feats_pack_T, output_single_T, accum_T, coords_diff_T, knn_dist_T, knn_idx_T,
                            exp_table_T, exp_table_idx_T, weighted_feature_T, CONFIG_T>(coords_forked[p], feats_forked[p],
                                                                                        res_forked[p], p);
    }

    join_outputs<P, output_single_T, output_pack_T, CONFIG_T>(res_forked, res_stream);
}

} // namespace nnet
#endif
