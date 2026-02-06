#include "nnet_gravnet_bitonic_sort.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

constexpr int SORT_N = 8;

struct bitonic_sort_array_params {
    std::vector<int> dist_in;
    std::vector<int> idx_in;
    std::vector<int> dist_expected;
    std::vector<int> idx_expected;
};

class Hls4mlGravNetBitonicSortArrayTest : public ::testing::TestWithParam<bitonic_sort_array_params> {};

TEST_P(Hls4mlGravNetBitonicSortArrayTest, SortsArbitrarySequenceCorrectly) {
    bitonic_sort_array_params params = GetParam();

    // Sanity check
    ASSERT_EQ(params.dist_in.size(), SORT_N);

    // Copy vectors to arrays
    int dist[SORT_N];
    int idx[SORT_N];
    std::copy(params.dist_in.begin(), params.dist_in.end(), dist);
    std::copy(params.idx_in.begin(), params.idx_in.end(), idx);

    nnet::bitonic_sort_array<SORT_N>(dist, idx);

    for (int i = 0; i < SORT_N; ++i) {
        EXPECT_EQ(dist[i], params.dist_expected[i]) << "Distance mismatch at index " << i;
        EXPECT_EQ(idx[i], params.idx_expected[i]) << "Index mismatch at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    SortScenariosN8, Hls4mlGravNetBitonicSortArrayTest,
    ::testing::Values(
        // Case 1: Random mixed order
        bitonic_sort_array_params{{80, 10, 50, 20, 70, 40, 60, 30},
                                  {8, 1, 5, 2, 7, 4, 6, 3}, // indices track values for easy verification
                                  {10, 20, 30, 40, 50, 60, 70, 80},
                                  {1, 2, 3, 4, 5, 6, 7, 8}},
        // Case 2: Fully Reversed
        bitonic_sort_array_params{
            {8, 7, 6, 5, 4, 3, 2, 1}, {0, 0, 0, 0, 0, 0, 0, 0}, {1, 2, 3, 4, 5, 6, 7, 8}, {0, 0, 0, 0, 0, 0, 0, 0}},
        // Case 3: Already Sorted
        bitonic_sort_array_params{{10, 20, 30, 40, 50, 60, 70, 80},
                                  {0, 1, 2, 3, 4, 5, 6, 7},
                                  {10, 20, 30, 40, 50, 60, 70, 80},
                                  {0, 1, 2, 3, 4, 5, 6, 7}}));
