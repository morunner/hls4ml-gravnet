#ifndef NNET_GRAVNET_CORE_STREAM_H_
#define NNET_GRAVNET_CORE_STREAM_H_

#include "hls_stream.h"
#include "nnet_gravnet_bitonic_sort.h"
#include "nnet_gravnet_core_common.h"
#include "nnet_types.h"

namespace nnet {

template <unsigned n_iterations, unsigned n_pack, class coords_T, class coords_diff_T, class knn_dist_T, class knn_idx_T,
          typename CONFIG_T>
void calculate_squared_distances_one_pack(
    typename coords_T::value_type coords_buffer[CONFIG_T::V][CONFIG_T::S],
    typename coords_T::value_type query_coords_buffer[n_iterations][CONFIG_T::S],
    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    unsigned int pack_idx) {

#pragma HLS INLINE off

    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1

        unsigned int global_u_idx = i * n_pack + pack_idx;

        nnet::array<knn_dist_T, CONFIG_T::n_neighbours> chunks_dist[CONFIG_T::V / CONFIG_T::n_neighbours];
        nnet::array<knn_idx_T, CONFIG_T::n_neighbours> chunks_idx[CONFIG_T::V / CONFIG_T::n_neighbours];
#pragma HLS ARRAY_PARTITION variable = chunks_dist complete
#pragma HLS ARRAY_PARTITION variable = chunks_idx complete

        typename coords_T::value_type query_coords[CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = query_coords complete
        for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
            query_coords[s] = query_coords_buffer[i][s];
        }

        for (unsigned int j = 0; j < CONFIG_T::V; j++) {
#pragma HLS UNROLL

            unsigned int L = j / CONFIG_T::n_neighbours;
            unsigned int k = j % CONFIG_T::n_neighbours;

            knn_dist_T dist_sq = 0;
            for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                coords_diff_T diff = query_coords[s] - coords_buffer[j][s];
                dist_sq += (knn_dist_T)(diff * diff);
            }

            if (global_u_idx == j) {
                dist_sq = 32000;
            }

            chunks_dist[L][k] = dist_sq;
            chunks_idx[L][k] = (knn_idx_T)j;
        }

        for (unsigned int L = 0; L < (CONFIG_T::V / CONFIG_T::n_neighbours); L++) {
#pragma HLS UNROLL
            dist_streams[L].write(chunks_dist[L]);
            idx_streams[L].write(chunks_idx[L]);
        }
    }
}

template <unsigned n_iterations, class knn_dist_T, class knn_idx_T, typename CONFIG_T>
void select_knn_one_pack(
    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[CONFIG_T::V / CONFIG_T::n_neighbours],
    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> &knn_dists,
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> &knn_indices) {

    constexpr unsigned num_leaves = CONFIG_T::V / CONFIG_T::n_neighbours;

    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1

        nnet::array<knn_dist_T, CONFIG_T::n_neighbours> current_best_d = dist_streams[0].read();
        nnet::array<knn_idx_T, CONFIG_T::n_neighbours> current_best_i = idx_streams[0].read();

        bitonic_sort_array<CONFIG_T::n_neighbours>(current_best_d, current_best_i);

        for (unsigned int n = 1; n < num_leaves; n++) {
#pragma HLS UNROLL
            nnet::array<knn_dist_T, CONFIG_T::n_neighbours> next_d = dist_streams[n].read();
            nnet::array<knn_idx_T, CONFIG_T::n_neighbours> next_i = idx_streams[n].read();

            bitonic_sort_array<CONFIG_T::n_neighbours>(next_d, next_i);

            nnet::array<knn_dist_T, CONFIG_T::n_neighbours> merged_d;
            nnet::array<knn_idx_T, CONFIG_T::n_neighbours> merged_i;

            merge_and_keep_k<CONFIG_T::n_neighbours>(current_best_d, current_best_i, next_d,
                                                                                   next_i, merged_d, merged_i);

            current_best_d = merged_d;
            current_best_i = merged_i;
        }

        knn_dists.write(current_best_d);
        knn_indices.write(current_best_i);
    }
}

template <unsigned n_iterations, class feat_val_T, class output_T, class accum_T, class knn_dist_T, class knn_idx_T,
          class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void apply_weights_and_reduce_one_pack(hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> &knn_dists,
                                       hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> &knn_indices,
                                       feat_val_T feats_buffer[CONFIG_T::V][CONFIG_T::F],
                                       exp_table_T exp_table[CONFIG_T::exp_table_size],
                                       hls::stream<output_T> &res_stream_local) {

    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1

        accum_T acc_sum[CONFIG_T::F];
        accum_T acc_max[CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = acc_sum complete
#pragma HLS ARRAY_PARTITION variable = acc_max complete

        for (int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            acc_sum[f] = 0;
            acc_max[f] = -32000;
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
                feat_val_T feat = feats_buffer[idx][f];
                weighted_feature_T val = (weighted_feature_T)(feat * w);
                acc_sum[f] += val;
                if (val > acc_max[f])
                    acc_max[f] = val;
            }
        }

        output_T out_local;
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            out_local[f] = acc_max[f];
            out_local[CONFIG_T::F + f] = acc_sum[f] / (accum_T)CONFIG_T::n_neighbours;
        }
        res_stream_local.write(out_local);
    }
}

template <unsigned n_iterations, unsigned n_pack, class coords_T, class feats_T, class output_T, class coord_val_T,
          class feat_val_T, class coords_diff_T, class knn_dist_T, class knn_idx_T, class exp_table_T, class exp_table_idx_T,
          class weighted_feature_T, class accum_T, typename CONFIG_T>
void gravnet_core_pack_wrapper(typename coords_T::value_type coords_buffer[CONFIG_T::V][CONFIG_T::S],
                               typename coords_T::value_type query_coords_buffer[n_iterations][CONFIG_T::S],
                               feat_val_T feats_buffer[CONFIG_T::V][CONFIG_T::F],
                               exp_table_T exp_table[CONFIG_T::exp_table_size], hls::stream<output_T> &res_stream_local,
                               unsigned int pack_idx) {

#pragma HLS DATAFLOW

    constexpr unsigned num_leaves = CONFIG_T::V / CONFIG_T::n_neighbours;

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> dist_streams[num_leaves];
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> idx_streams[num_leaves];
#pragma HLS STREAM variable = dist_streams depth = 4
#pragma HLS STREAM variable = idx_streams depth = 4

    hls::stream<nnet::array<knn_dist_T, CONFIG_T::n_neighbours>> knn_dists;
    hls::stream<nnet::array<knn_idx_T, CONFIG_T::n_neighbours>> knn_indices;
#pragma HLS STREAM variable = knn_dists depth = 4
#pragma HLS STREAM variable = knn_indices depth = 4

    calculate_squared_distances_one_pack<n_iterations, n_pack, coords_T, coords_diff_T, knn_dist_T, knn_idx_T, CONFIG_T>(
        coords_buffer, query_coords_buffer, dist_streams, idx_streams, pack_idx);

    select_knn_one_pack<n_iterations, knn_dist_T, knn_idx_T, CONFIG_T>(dist_streams, idx_streams, knn_dists, knn_indices);

    apply_weights_and_reduce_one_pack<n_iterations, feat_val_T, output_T, accum_T, knn_dist_T, knn_idx_T, exp_table_T,
                                      exp_table_idx_T, weighted_feature_T, CONFIG_T>(knn_dists, knn_indices, feats_buffer,
                                                                                     exp_table, res_stream_local);
}

template <class coords_T, class feats_T, class output_T, class accum_T, class coords_diff_T, class knn_dist_T,
          class knn_idx_T, class exp_table_T, class exp_table_idx_T, class weighted_feature_T, typename CONFIG_T>
void gravnet_core(hls::stream<coords_T> &coords_stream, hls::stream<feats_T> &feats_stream,
                  hls::stream<output_T> &res_stream) {
#pragma HLS DATAFLOW

    constexpr unsigned n_pack = coords_T::size / CONFIG_T::S;
    constexpr unsigned n_iterations = CONFIG_T::V / n_pack;

    typedef typename coords_T::value_type coord_val_t;
    typedef typename feats_T::value_type feat_val_t;

    coord_val_t coords_buffer[CONFIG_T::V][CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = coords_buffer complete dim = 0
    feat_val_t feats_buffer[CONFIG_T::V][CONFIG_T::F];
#pragma HLS ARRAY_PARTITION variable = feats_buffer complete dim = 0
    static exp_table_T exp_table[CONFIG_T::exp_table_size];
#pragma HLS ARRAY_PARTITION variable = exp_table complete
    gravnet_init_exp_table<exp_table_T, CONFIG_T>(exp_table);

    coord_val_t query_coords_per_pack[n_pack][n_iterations][CONFIG_T::S];
#pragma HLS ARRAY_PARTITION variable = query_coords_per_pack complete dim = 1

    typedef nnet::array<typename output_T::value_type, 2 * CONFIG_T::F> local_out_t;
    hls::stream<local_out_t> local_res_streams[n_pack];
#pragma HLS STREAM variable = local_res_streams depth = 4

    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1
        coords_T c_pack = coords_stream.read();
        feats_T f_pack = feats_stream.read();

        for (unsigned int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL
            unsigned int v_idx = i * n_pack + p;
            for (unsigned int s = 0; s < CONFIG_T::S; s++) {
#pragma HLS UNROLL
                coords_buffer[v_idx][s] = c_pack[p * CONFIG_T::S + s];
                query_coords_per_pack[p][i][s] = c_pack[p * CONFIG_T::S + s];
            }
            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
                feats_buffer[v_idx][f] = f_pack[p * CONFIG_T::F + f];
            }
        }
    }

    for (unsigned int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL
        gravnet_core_pack_wrapper<n_iterations, n_pack, coords_T, feats_T, local_out_t, coord_val_t, feat_val_t,
                                  coords_diff_T, knn_dist_T, knn_idx_T, exp_table_T, exp_table_idx_T, weighted_feature_T,
                                  accum_T, CONFIG_T>(coords_buffer, query_coords_per_pack[p], feats_buffer, exp_table,
                                                     local_res_streams[p], p);
    }

    for (unsigned int i = 0; i < n_iterations; i++) {
#pragma HLS PIPELINE II = 1
        output_T out_pack;

        for (unsigned int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL
            local_out_t local_res = local_res_streams[p].read();
            unsigned int offset = p * 2 * CONFIG_T::F;
            for (unsigned int f = 0; f < 2 * CONFIG_T::F; f++) {
#pragma HLS UNROLL
                out_pack[offset + f] = local_res[f];
            }
        }
        res_stream.write(out_pack);
    }
}

} // namespace nnet
#endif
