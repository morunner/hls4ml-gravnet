#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_global_exchange.h"
#include "nnet_gravnet_core.h"
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
        result_t actual_result[gravnet_config::V * 4 * gravnet_config::F];
        nnet::global_exchange<x_t, result_t, mean_t, gravnet_config>(x, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, gravnet_core) {
    const double abs_error = 0.006;
    typedef ap_fixed<16, 8> input_t;
    typedef ap_fixed<16, 8> output_t;
    typedef ap_fixed<16, 8> knn_dist_t;
    typedef unsigned int knn_idx_t;
    typedef ap_ufixed<16, 0> exp_t;
    typedef ap_fixed<16, 8> weight_t;

    for (size_t i = 0; i < gravnet_core_test_vectors_length; i++) {
        gravnet_core_test_vector v = gravnet_core_test_vectors[i];

        input_t coords[v.coords_len];
        input_t feats[v.feats_len];
        for (unsigned int i = 0; i < v.coords_len; i++) {
            coords[i] = v.coords[i];
        }
        for (unsigned int i = 0; i < v.feats_len; i++) {
            feats[i] = v.feats[i];
        }

        output_t actual_result[gravnet_config::V * 2 * gravnet_config::F];
        nnet::gravnet_core<input_t, output_t, knn_dist_t, knn_idx_t, exp_t, weight_t, gravnet_config>(coords, feats,
                                                                                                      actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}
