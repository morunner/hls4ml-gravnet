#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "nnet_global_exchange.h"
#include "nnet_global_exchange_stream.h"
#include "nnet_gravnet_core.h"
#include "nnet_gravnet_core_stream.h"
#include "nnet_types.h"
#include "test_vectors.h"
#include "gtest/gtest.h"
#include <cstddef>

TEST(Hls4mlGravNetTest, global_exchange) {
    const double abs_error = 0.006;
    typedef ap_fixed<16, 8> x_t;
    typedef ap_fixed<16, 8> mean_t;
    typedef ap_fixed<16, 8> result_t;

    for (size_t i = 0; i < global_exchange_test_vectors_length; i++) {
        global_exchange_test_vector v = global_exchange_test_vectors[i];

        x_t x[v.x_len];
        for (unsigned int i = 0; i < v.x_len; i++) {
            x[i] = v.x[i];
        }
        result_t actual_result[global_exchange_config::V * 4 * global_exchange_config::F];

        // Array version uses scalar types for template args
        nnet::global_exchange<x_t, result_t, mean_t, global_exchange_config>(x, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, global_exchange_stream) {
    const double abs_error = 0.006;
    typedef ap_fixed<16, 8> x_t;
    typedef ap_fixed<16, 8> mean_t;
    typedef ap_fixed<16, 8> result_t;

    const unsigned V = global_exchange_config::V;
    const unsigned F = global_exchange_config::F;

    typedef nnet::array<x_t, F> input_pack_t;
    typedef nnet::array<result_t, 4 * F> output_pack_t;

    for (size_t i = 0; i < global_exchange_test_vectors_length; i++) {
        global_exchange_test_vector v = global_exchange_test_vectors[i];

        hls::stream<input_pack_t> input_stream("input_stream");
        hls::stream<output_pack_t> output_stream("output_stream");

        for (unsigned int vertex = 0; vertex < V; vertex++) {
            input_pack_t vertex_data;
            for (unsigned int feature = 0; feature < F; feature++) {
                vertex_data[feature] = (x_t)v.x[vertex * F + feature];
            }
            input_stream.write(vertex_data);
        }

        nnet::global_exchange<input_pack_t, output_pack_t, mean_t, global_exchange_config>(input_stream, output_stream);

        result_t actual_result[V * 4 * F];

        for (unsigned int vertex = 0; vertex < V; vertex++) {
            output_pack_t res_data = output_stream.read();

            for (unsigned int feature = 0; feature < 4 * F; feature++) {
                actual_result[vertex * (4 * F) + feature] = res_data[feature];
            }
        }

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, gravnet_core) {
    const double abs_error = 0.01;

    assert(gravnet_config::exp_table_size == 32);
    assert(gravnet_config::V == 128);

    typedef ap_fixed<16, 8> coords_t;
    typedef ap_fixed<16, 8> feats_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> output_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> accum_t;
    typedef ap_fixed<16, 8> coords_diff_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> knn_dist_t;
    typedef ap_int<8> knn_idx_t;
    typedef ap_ufixed<8, 1> exp_table_t;
    typedef ap_uint<5> exp_table_idx_t;
    typedef ap_fixed<16, 8> feats_t;

    for (size_t i = 0; i < gravnet_core_test_vectors_length; i++) {
        gravnet_core_test_vector v = gravnet_core_test_vectors[i];

        coords_t coords[v.coords_len];
        feats_t feats[v.feats_len];
        for (unsigned int i = 0; i < v.coords_len; i++) {
            coords[i] = v.coords[i];
        }
        for (unsigned int i = 0; i < v.feats_len; i++) {
            feats[i] = v.feats[i];
        }

        output_t actual_result[gravnet_config::V * 2 * gravnet_config::F];
        nnet::gravnet_core<coords_t, feats_t, output_t, accum_t, coords_diff_t, knn_dist_t, knn_idx_t, exp_table_t,
                           exp_table_idx_t, feats_t, gravnet_config>(coords, feats, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, gravnet_core_stream) {
    const double abs_error = 0.01;

    const unsigned V = gravnet_config::V;
    const unsigned S = gravnet_config::S;
    const unsigned F = gravnet_config::F;

    typedef ap_fixed<16, 8> coords_t;
    typedef ap_fixed<16, 8> feats_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> output_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> accum_t;
    typedef ap_fixed<16, 8> coords_diff_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> knn_dist_t;
    typedef ap_int<8> knn_idx_t;
    typedef ap_ufixed<8, 1> exp_table_t;
    typedef ap_uint<5> exp_table_idx_t;
    typedef ap_fixed<16, 8> weighted_feature_t;

    typedef nnet::array<coords_t, S> coords_pack_t;
    typedef nnet::array<feats_t, F> feats_pack_t;
    typedef nnet::array<output_t, 2 * F> output_pack_t;

    for (size_t i = 0; i < gravnet_core_test_vectors_length; i++) {
        gravnet_core_test_vector v = gravnet_core_test_vectors[i];

        hls::stream<coords_pack_t> coords_stream("coords_stream");
        hls::stream<feats_pack_t> feats_stream("feats_stream");
        hls::stream<output_pack_t> res_stream("res_stream");

        for (unsigned int vertex = 0; vertex < V; vertex++) {
            coords_pack_t c_pack;
            for (unsigned int s = 0; s < S; s++) {
                c_pack[s] = (coords_t)v.coords[vertex * S + s];
            }
            coords_stream.write(c_pack);

            feats_pack_t f_pack;
            for (unsigned int f = 0; f < F; f++) {
                f_pack[f] = (feats_t)v.feats[vertex * F + f];
            }
            feats_stream.write(f_pack);
        }

        nnet::gravnet_core<coords_pack_t, feats_pack_t, output_pack_t, accum_t, coords_diff_t, knn_dist_t, knn_idx_t,
                           exp_table_t, exp_table_idx_t, weighted_feature_t, gravnet_config>(coords_stream, feats_stream,
                                                                                             res_stream);

        output_t actual_result[V * 2 * F];

        for (unsigned int vertex = 0; vertex < V; vertex++) {
            if (res_stream.empty()) {
                FAIL() << "Output stream is empty before reading all vertices at index " << vertex;
            }

            output_pack_t res_pack = res_stream.read();

            for (unsigned int f = 0; f < 2 * F; f++) {
                actual_result[vertex * (2 * F) + f] = res_pack[f];
            }
        }

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error) << "Mismatch at index " << j;
        }
    }
}
