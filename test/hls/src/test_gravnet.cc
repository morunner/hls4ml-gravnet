#include "gravnet.h"
#include "test_vectors.h"
#include "gtest/gtest.h"
#include <cstddef>

TEST(Hls4mlGravNetTest, global_exchange) {
    const double abs_error = 1e-6;

    for (size_t i = 0; i < global_exchange_test_vectors_length; i++) {
        global_exchange_test_vector v = global_exchange_test_vectors[i];

        float actual_result[gravnet_config::B * gravnet_config::V * 4 * gravnet_config::F];
        global_exchange<float, float, gravnet_config>(v.x, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, euclidean_squared) {
    const double abs_error = 1e-4;

    for (size_t i = 0; i < euclidean_squared_test_vectors_length; i++) {
        euclidean_squared_test_vector v = euclidean_squared_test_vectors[i];

        float actual_result[gravnet_config::B * gravnet_config::V * gravnet_config::V];
        euclidean_squared<float, float, gravnet_config>(v.A, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}

TEST(Hls4mlGravNetTest, euclidean_squared_knn) {
    const double abs_error = 1e-4;

    for (size_t i = 0; i < euclidean_squared_knn_test_vectors_length; i++) {
        euclidean_squared_knn_test_vector v = euclidean_squared_knn_test_vectors[i];

        float actual_ranked_distances[gravnet_config::B * gravnet_config::V * gravnet_config::n_neighbours];
        unsigned int actual_ranked_indices[gravnet_config::B * gravnet_config::V * gravnet_config::n_neighbours];
        euclidean_squared_knn<float, float, unsigned int, gravnet_config>(v.A, actual_ranked_distances,
                                                                          actual_ranked_indices);

        ASSERT_EQ(v.expected_ranked_distances_len, v.expected_ranked_indices_len);
        ASSERT_EQ(v.expected_ranked_distances_len, sizeof(actual_ranked_distances) / sizeof(actual_ranked_distances[0]));
        ASSERT_EQ(v.expected_ranked_indices_len, sizeof(actual_ranked_indices) / sizeof(actual_ranked_indices[0]));

        for (size_t j = 0; j < v.expected_ranked_indices_len; j++) {
            EXPECT_NEAR(v.expected_ranked_distances[j], actual_ranked_distances[j], abs_error);
            EXPECT_EQ(v.expected_ranked_indices[j], actual_ranked_indices[j]);
        }
    }
}

TEST(Hls4mlGravNetTest, gravnet_core) {
    const double abs_error = 1e-4;

    for (size_t i = 0; i < gravnet_core_test_vectors_length; i++) {
        gravnet_core_test_vector v = gravnet_core_test_vectors[i];

        float actual_result[gravnet_config::B * gravnet_config::V * 2 * gravnet_config::F];
        gravnet_core<float, float, unsigned int, float, float, float, gravnet_config>(v.coords, v.feats, actual_result);

        ASSERT_EQ(v.expected_result_len, sizeof(actual_result) / sizeof(actual_result[0]));

        for (size_t j = 0; j < v.expected_result_len; j++) {
            EXPECT_NEAR(v.expected_result[j], actual_result[j], abs_error);
        }
    }
}
