#include "nnet_gravnet_bitonic_sort.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

constexpr int KEEP_K = 8;

struct merge_k_params {
    std::vector<int> distA;
    std::vector<int> idxA;
    std::vector<int> distB;
    std::vector<int> idxB;
    std::vector<int> distOut_expected;
    std::vector<int> idxOut_expected;
};

class Hls4mlGravNetMergeAndKeepKTest : public ::testing::TestWithParam<merge_k_params> {};

TEST_P(Hls4mlGravNetMergeAndKeepKTest, MergesAndKeepsSmallestK) {
    merge_k_params params = GetParam();

    // Sanity check
    ASSERT_EQ(params.distA.size(), KEEP_K);
    ASSERT_EQ(params.distB.size(), KEEP_K);

    // Vectors -> C arrays
    int distA[KEEP_K], idxA[KEEP_K];
    int distB[KEEP_K], idxB[KEEP_K];
    int distOut[KEEP_K], idxOut[KEEP_K];

    std::copy(params.distA.begin(), params.distA.end(), distA);
    std::copy(params.idxA.begin(), params.idxA.end(), idxA);
    std::copy(params.distB.begin(), params.distB.end(), distB);
    std::copy(params.idxB.begin(), params.idxB.end(), idxB);

    merge_and_keep_k<KEEP_K, int, int>(distA, idxA, distB, idxB, distOut, idxOut);

    for (int i = 0; i < KEEP_K; ++i) {
        EXPECT_EQ(distOut[i], params.distOut_expected[i]) << "Output dist mismatch at index " << i;
        EXPECT_EQ(idxOut[i], params.idxOut_expected[i]) << "Output idx mismatch at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(MergeKeepKScenariosN8, Hls4mlGravNetMergeAndKeepKTest,
                         ::testing::Values(
                             // Case 1: Interleaved inputs
                             // A = [10, 30, 50, 70, 90, 110, 130, 150]
                             // B = [20, 40, 60, 80, 100, 120, 140, 160]
                             // Expected Top 8: [10, 20, 30, 40, 50, 60, 70, 80]
                             merge_k_params{{10, 30, 50, 70, 90, 110, 130, 150},
                                            {1, 3, 5, 7, 9, 11, 13, 15},
                                            {20, 40, 60, 80, 100, 120, 140, 160},
                                            {2, 4, 6, 8, 10, 12, 14, 16},
                                            {10, 20, 30, 40, 50, 60, 70, 80},
                                            {1, 2, 3, 4, 5, 6, 7, 8}},

                             // Case 2: A is completely smaller than B
                             // A = [1..8], B = [101..108]
                             // Expected Top 8: [1..8] (Copy of A)
                             merge_k_params{{1, 2, 3, 4, 5, 6, 7, 8},
                                            {0, 0, 0, 0, 0, 0, 0, 0},
                                            {101, 102, 103, 104, 105, 106, 107, 108},
                                            {1, 1, 1, 1, 1, 1, 1, 1},
                                            {1, 2, 3, 4, 5, 6, 7, 8},
                                            {0, 0, 0, 0, 0, 0, 0, 0}},

                             // Case 3: Mixed overlap
                             // A = [5, 15, 25, 35, 45, 55, 65, 75]
                             // B = [1, 2, 3, 4, 6, 7, 8, 9]
                             // Combined Sorted: 1, 2, 3, 4, 5, 6, 7, 8, 9, 15...
                             // Expected Top 8: [1, 2, 3, 4, 5, 6, 7, 8]
                             merge_k_params{{5, 15, 25, 35, 45, 55, 65, 75},
                                            {100, 0, 0, 0, 0, 0, 0, 0},
                                            {1, 2, 3, 4, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 6, 7, 8, 9},
                                            {1, 2, 3, 4, 5, 6, 7, 8},
                                            {1, 2, 3, 4, 100, 6, 7, 8}}));
