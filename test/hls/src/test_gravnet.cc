#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_gravnet_core.h"
#include "nnet_global_exchange.h"
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
    const double abs_error = 0.1;

    // Ensure the right exp_table_idx_t, knn_idx_t bitwidths are set
    assert(gravnet_config::exp_table_size == 32);
    assert(gravnet_config::V == 128);

    typedef ap_fixed<16, 8> coords_t;
    typedef ap_fixed<16, 8> feats_t;
    typedef ap_fixed<16, 8, AP_RND, AP_SAT> output_t;
    typedef ap_fixed<32, 16> knn_dist_t;
    typedef ap_uint<7> knn_idx_t;
    typedef ap_ufixed<16, 1> exp_table_t;
    typedef ap_uint<5> exp_table_idx_t;
    typedef ap_fixed<16, 8> weight_t;

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
        nnet::gravnet_core<coords_t, feats_t, output_t, knn_dist_t, knn_idx_t, exp_table_t, exp_table_idx_t, weight_t,
                           gravnet_config>(coords, feats, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}
