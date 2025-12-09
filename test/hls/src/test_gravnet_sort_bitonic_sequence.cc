#include "nnet_gravnet_bitonic_sort.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

constexpr int TEST_N = 8;

struct sort_bitonic_sequence_params {
    std::vector<int> dist_in;
    std::vector<int> idx_in;
    std::vector<int> dist_expected;
    std::vector<int> idx_expected;
};

class Hls4mlGravNetSortBitonicSequenceTest : public ::testing::TestWithParam<sort_bitonic_sequence_params> {};

TEST_P(Hls4mlGravNetSortBitonicSequenceTest, SortsBitonicSequenceCorrectly) {
    sort_bitonic_sequence_params params = GetParam();

    // Sanity check
    ASSERT_EQ(params.dist_in.size(), TEST_N) << "Input dist vector size must match TEST_N";
    ASSERT_EQ(params.idx_in.size(), TEST_N) << "Input idx vector size must match TEST_N";

    // Convert vector to C style array
    int dist[TEST_N];
    int idx[TEST_N];
    std::copy(params.dist_in.begin(), params.dist_in.end(), dist);
    std::copy(params.idx_in.begin(), params.idx_in.end(), idx);

    // Call the function under test
    sort_bitonic_sequence<TEST_N, int, int>(dist, idx);

    // Verify results
    for (int i = 0; i < TEST_N; ++i) {
        EXPECT_EQ(dist[i], params.dist_expected[i]) << "Distance mismatch at index " << i;
        EXPECT_EQ(idx[i], params.idx_expected[i]) << "Index mismatch at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(MergeScenarios, Hls4mlGravNetSortBitonicSequenceTest,
                         ::testing::Values(
                             // Case 1: Already Sorted (Ascending part of bitonic sequence)
                             sort_bitonic_sequence_params{{10, 20, 30, 40, 50, 60, 70, 80},
                                                          {0, 1, 2, 3, 4, 5, 6, 7},
                                                          {10, 20, 30, 40, 50, 60, 70, 80},
                                                          {0, 1, 2, 3, 4, 5, 6, 7}},

                             // Case 2: Reverse Sorted (Descending part of bitonic sequence)
                             sort_bitonic_sequence_params{{80, 70, 60, 50, 40, 30, 20, 10},
                                                          {0, 1, 2, 3, 4, 5, 6, 7},
                                                          {10, 20, 30, 40, 50, 60, 70, 80},
                                                          {7, 6, 5, 4, 3, 2, 1, 0}},

                             // Case 3: Mountain (Ascending then Descending)
                             // [10, 30, 50, 70, 80, 60, 40, 20]
                             sort_bitonic_sequence_params{{10, 30, 50, 70, 80, 60, 40, 20},
                                                          {0, 1, 2, 3, 4, 5, 6, 7},
                                                          {10, 20, 30, 40, 50, 60, 70, 80},
                                                          {0, 7, 1, 6, 2, 5, 3, 4}},

                             // Case 4: Valley (Descending then Ascending)
                             // [80, 60, 40, 20, 10, 30, 50, 70]
                             sort_bitonic_sequence_params{{80, 60, 40, 20, 10, 30, 50, 70},
                                                          {0, 1, 2, 3, 4, 5, 6, 7},
                                                          {10, 20, 30, 40, 50, 60, 70, 80},
                                                          {4, 3, 5, 2, 6, 1, 7, 0}}));
