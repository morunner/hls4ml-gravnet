/**
 * @brief Swaps two given pairs (distance, index)
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
 * @brief Compares two elements (distance, index) and assigns the pair with the smaller distance to the first element
 */
template <typename dist_T, typename idx_T> void gravnet_compare_and_swap(dist_T &d1, idx_T &i1, dist_T &d2, idx_T &i2) {
#pragma HLS INLINE
    if (d2 < d1) {
        gravnet_swap(d1, i1, d2, i2);
    }
}

/**
 * @brief Sorts a bitonic sequence by dist and keeps track of the corresponding index idx by comparing the corresponding
 * elements in the ascending and descending part
 */
template <int N, typename dist_T, typename idx_T> void sort_bitonic_sequence(dist_T dist[N], idx_T idx[N]) {
#pragma HLS INLINE
    for (int stride = N / 2; stride > 0; stride /= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < N; i++) {
#pragma HLS UNROLL
            if ((i % (2 * stride)) < stride) {
                gravnet_compare_and_swap(dist[i], idx[i], dist[i + stride], idx[i + stride]);
            }
        }
    }
}

/**
 * @brief Sorts a given array using bitonic sort and keeps track of the corresponding index
 */
template <int N, typename dist_T, typename idx_T> void bitonic_sort_array(dist_T dist[N], idx_T idx[N]) {
#pragma HLS INLINE
    for (int seq_size = 2; seq_size <= N; seq_size *= 2) {
#pragma HLS UNROLL
        for (int i = 0; i < N; i += seq_size) {
#pragma HLS UNROLL

            // Create a bitonic sequence from two lists
            //  This also includes the first merge (see below) to avoid swapping twice
            int start = i;
            int end = i + seq_size - 1;
            for (int k = 0; k < seq_size / 2; k++) {
#pragma HLS UNROLL
                int idx1 = start + k;
                int idx2 = end - k;
                gravnet_compare_and_swap(dist[idx1], idx[idx1], dist[idx2], idx[idx2]);
            }

            // Perform bitonic merge
            //  Compare the element of the smaller part (left) with the corresponding element of the larger part
            //  (right) of the sequence. First sequence has already been swapped above.
            for (int stride = seq_size / 4; stride > 0; stride /= 2) {
#pragma HLS UNROLL
                for (int j = 0; j < seq_size; j++) {
#pragma HLS UNROLL
                    if ((j % (2 * stride)) < stride) {
                        int pos = i + j;
                        gravnet_compare_and_swap(dist[pos], idx[pos], dist[pos + stride], idx[pos + stride]);
                    }
                }
            }
        }
    }
}

/**
 * @brief Bitonically merges two sorted lists and keeps the top-k elements
 */
template <int K, typename dist_T, typename idx_T>
void merge_and_keep_k(dist_T left_dists[K], idx_T left_indices[K], dist_T right_dists[K], idx_T right_indices[K],
                      dist_T dist_out[K], idx_T idx_out[K]) {
#pragma HLS INLINE

#pragma HLS ARRAY_PARTITION variable = dist_out complete
#pragma HLS ARRAY_PARTITION variable = idx_out complete

    // We only care about the top-k elements
    for (int i = 0; i < K; i++) {
#pragma HLS UNROLL
        dist_T ld = left_dists[i];
        idx_T li = left_indices[i];

        // right array is reverted to allow bitonic comparison
        dist_T rd = right_dists[K - 1 - i];
        idx_T ri = right_indices[K - 1 - i];

        if (rd < ld) {
            dist_out[i] = rd;
            idx_out[i] = ri;
        } else {
            dist_out[i] = ld;
            idx_out[i] = li;
        }
    }

    // Sort Result
    sort_bitonic_sequence<K, dist_T, idx_T>(dist_out, idx_out);
}
