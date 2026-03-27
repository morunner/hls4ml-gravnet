/**
 * @file nnet_gravnet_bitonic_sort.h
 * @brief Hardware-optimized bitonic sorting networks for High-Level Synthesis (HLS).
 * * This file contains functionality to sort arrays/sequences using a bitonic sorting network
 * initially proposed by:
 * Batcher, Kenneth E. "Sorting networks and their applications." Proceedings of the
 * April 30--May 2, 1968, spring joint computer conference. 1968.
 * * The lecture slides on parallel sorting (27 May 2020) from the "Parallel Programming"
 * course held in spring 2020 at ETH Zurich greatly helped in understanding these networks:
 * https://spcl.inf.ethz.ch/Teaching/2020-pp/
 */

#ifndef NNET_GRAVNET_BITONIC_SORT_H_
#define NNET_GRAVNET_BITONIC_SORT_H_

namespace nnet {

/**
 * @brief Swaps two elements using their references.
 */
template <typename dist_T, typename idx_T> void gravnet_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    dist_T temp_d = d1;
    d1 = d2;
    d2 = temp_d;
    idx_T temp_i = i1;
    i1 = i2;
    i2 = temp_i;
}

/**
 * @brief Compare-and-swap: Swaps d2 and d1 (and their indices) if d2 < d1.
 */
template <typename dist_T, typename idx_T> void gravnet_compare_and_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    if (d2 < d1) {
        gravnet_swap(d1, i1, d2, i2);
    }
}

/**
 * @brief Performs a full bitonic sort on an unsorted input array.
 *
 * - First, only adjacent elements d1, d2 are compared and swapped if d2 < d1, making up sorted arrays of length 2.
 * - Then, expand: Two adjacent arrays of length two form a bitonic sequence by implicitly reverting the right array.
 * - These can then be sorted again using the bitonic merge step.
 * - The previous two steps are repeated until the entire array is sorted.
 *
 * Here is an example for a size 8 bitonic sorting network:
 *
 * 7 ---o--- 3 ---o------- 3 ---o--- 3 ---o--------------- 3 ---o------- 2 ---o--- 1
 * |         |             |         |                     |             |
 * 3 ---o--- 7 ---|---o--- 5 ---o--- 5 ---|---o----------- 4 ---|---o--- 1 ---o--- 2
 * |   |                   |   |                 |   |         |
 * 5 ---o--- 5 ---|---o--- 7 ---o--- 7 ---|---|---o------- 2 ---o---|--- 3 ---o--- 3
 * |         |             |         |   |   |             |   |         |
 * 8 ---o--- 8 ---o------- 8 ---o--- 8 ---|---|---|---o--- 1 ---|---o--- 4 ---o--- 4
 * |   |   |   |         |             |
 * 1 ---o--- 1 ---o------- 1 ---o--- 1 ---|---|---|---o--- 8 ---|---o--- 5 ---o--- 5
 * |         |             |         |   |   |             |   |         |
 * 6 ---o--- 6 ---|---o--- 2 ---o--- 2 ---|---|---o------- 7 ---o---|--- 6 ---o--- 6
 * |   |                   |   |                 |   |         |
 * 2 ---o--- 2 ---|---o--- 6 ---o--- 4 ---|---o----------- 5 ---|---o--- 8 ---o--- 7
 * |         |             |         |                     |             |
 * 4 ---o--- 4 ---o------- 4 ---o--- 6 ---o--------------- 6 ---o------- 7 ---o--- 8
 */
template <int K, typename dist_T, typename idx_T> void bitonic_sort_array(dist_T &dist, idx_T &idx) {
#pragma HLS INLINE

    // The current length of the bitonic sequences to sort.
    // Start with sequences of length two, which are sorted by a single compare and swap.
    for (int seq_size = 2; seq_size <= K; seq_size *= 2) {
#pragma HLS UNROLL
        // Process each subsequence
        for (int start = 0; start < K; start += seq_size) {
#pragma HLS UNROLL
            int end = start + seq_size - 1;

            // Single binary split: This splits one bitonic sequence into two, where all values in
            // the resulting left half are smaller than or equal to the values in the right one.
            for (int k = 0; k < seq_size / 2; k++) {
#pragma HLS UNROLL
                // Implicitly revert the right array to form a bitonic sequence from two adjacent arrays.
                int idx1 = start + k;
                int idx2 = end - k;

                // Compare and swap moves smaller values to the left and larger ones to the right.
                gravnet_compare_and_swap(dist[idx1], idx[idx1], dist[idx2], idx[idx2]);
            }

            // Bitonic merger, which sorts a bitonic sequence
            for (int stride = seq_size / 4; stride > 0; stride /= 2) {
#pragma HLS UNROLL
                for (int j = 0; j < seq_size; j++) {
#pragma HLS UNROLL
                    if ((j % (2 * stride)) < stride) {
                        int pos = start + j;
                        gravnet_compare_and_swap(dist[pos], idx[pos], dist[pos + stride], idx[pos + stride]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Sorts an already existing bitonic sequence.
 * * Sorts a bitonic sequence by iteratively applying bitonic splits.
 * Since the input sequence is assumed to already be bitonic, the right half
 * does not need to be accessed in reverse order (as opposed to the full sort function above)
 */
template <int K, typename dist_T, typename idx_T> void sort_bitonic_sequence(dist_T &dist, idx_T &idx) {
#pragma HLS INLINE
    for (int stride = K / 2; stride > 0; stride /= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
            if ((i % (2 * stride)) < stride) {
                gravnet_compare_and_swap(dist[i], idx[i], dist[i + stride], idx[i + stride]);
            }
        }
    }
}

/**
 * @brief Merges two sorted arrays and truncates to the K nearest neighbors.
 * * Takes two sorted arrays and merges them through bitonic compare and swap operations:
 * - The right array is accessed in reverse order such that left and right form a bitonic sequence.
 * - Compares elements of the given arrays at their corresponding indices.
 * - The smaller element is moved to the left and the larger to the right array.
 * - Due to the bitonic split property, all elements in the resulting left array are
 *   smaller than or equal to the ones in the right array.
 * - Hence, only the left array contains viable candidates for nearest neighbours.
 * - The left array is passed to the next tree level (and fully sorted), whereas the right one is discarded.
 */
template <int K, typename T_Arr_D, typename T_Arr_I>
void merge_and_keep_k(T_Arr_D &left_dists, T_Arr_I &left_indices, T_Arr_D &right_dists, T_Arr_I &right_indices,
                      T_Arr_D &dist_out, T_Arr_I &idx_out) {
#pragma HLS INLINE

// Completely partition for parallel access
#pragma HLS ARRAY_PARTITION variable = dist_out complete
#pragma HLS ARRAY_PARTITION variable = idx_out complete

    for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
        dist_out[i] = (right_dists[K - 1 - i] < left_dists[i]) ? right_dists[K - 1 - i] : left_dists[i];
        idx_out[i] = (right_dists[K - 1 - i] < left_dists[i]) ? right_indices[K - 1 - i] : left_indices[i];
    }
    sort_bitonic_sequence<K>(dist_out, idx_out);
}

} // namespace nnet

#endif
