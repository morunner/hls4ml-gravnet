#include "nnet_gravnet_bitonic_sort.h"

#include <gtest/gtest.h>

struct compare_and_swap_params {
    int d1_in;
    int i1_in;
    int d2_in;
    int i2_in;
    int d1_expected;
    int i1_expected;
    int d2_expected;
    int i2_expected;
};

class Hls4mlGravNetCompareAndSwapTest : public ::testing::TestWithParam<compare_and_swap_params> {};

TEST_P(Hls4mlGravNetCompareAndSwapTest, SwapsCorrectlyBasedOnDistance) {
    compare_and_swap_params params = GetParam();

    gravnet_compare_and_swap(params.d1_in, params.i1_in, params.d2_in, params.i2_in);

    EXPECT_EQ(params.d1_in, params.d1_expected) << "d1 (smaller) mismatch";
    EXPECT_EQ(params.d2_in, params.d2_expected) << "d2 (larger) mismatch";
    EXPECT_EQ(params.i1_in, params.i1_expected) << "i1 index mismatch";
    EXPECT_EQ(params.i2_in, params.i2_expected) << "i2 index mismatch";
}

INSTANTIATE_TEST_SUITE_P(SwapScenarios, Hls4mlGravNetCompareAndSwapTest,
                         ::testing::Values(
                             // Case 1: Already sorted (d1 < d2)
                             // Input: (10, 20) -> Expected: (10, 20)
                             // No swap should occur.
                             compare_and_swap_params{10, 1, 20, 2, 10, 1, 20, 2},

                             // Case 2: Unsorted (d1 > d2)
                             // Input: (50, 10) -> Expected: (10, 50)
                             // A swap should occur to order them by distance.
                             compare_and_swap_params{50, 5, 10, 1, 10, 1, 50, 5},

                             // Case 3: Equal distances
                             // Input: (5, 5) -> Expected: (5, 5)
                             // Stability check: If distances are equal, they should ideally remain (or swap is irrelevant),
                             // but here we expect them to remain in input order.
                             compare_and_swap_params{5, 100, 5, 200, 5, 100, 5, 200},

                             // Case 4: Negative values
                             // Input: (-1, -5) -> Expected: (-5, -1)
                             // Verifies the logic handles signed integer comparisons correctly.
                             compare_and_swap_params{-1, 1, -5, 2, -5, 2, -1, 1}));
