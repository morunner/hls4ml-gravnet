#include "ap_int.h"
#include <cmath>
#include <sys/types.h>

// Global Exchange

/**
 * @brief Performs global exchange by calculating mean, min, and max for each feature across all vertices.
 *
 * For each feature, this function computes its mean, minimum, and maximum value across all vertices
 * within a batch. The result for each vertex is a concatenation of the computed mean, min, max, and
 * the original input feature value.
 *
 * @tparam input_T The data type of the input features.
 * @tparam output_T The data type of the output features.
 * @tparam mean_T The data type for mean calculation.
 * @tparam CONFIG_T The configuration struct containing dimensions like B (batch size), V (vertices), and F (features).
 * @param x Input data array of shape (B, V, F), flattened.
 * @param res Output data array of shape (B, V, 4*F), flattened. For each vertex, the output is
 *            [mean(F_i), min(F_i), max(F_i), x_i] for each feature i.
 */
template <class input_T, class output_T, class mean_T, typename CONFIG_T>
void global_exchange(input_T x[CONFIG_T::B * CONFIG_T::V * CONFIG_T::F],
                     output_T res[CONFIG_T::B * CONFIG_T::V * 4 * CONFIG_T::F]) {
    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
            unsigned int first_v_index = b * (CONFIG_T::V * CONFIG_T::F) + f;
            input_T first_val = x[first_v_index];

            // Initialize
            input_T current_mean = 0;
            input_T current_min = first_val;
            input_T current_max = first_val;

            // Calculate mean, min, max
            for (unsigned int v = 0; v < CONFIG_T::V; v++) {
                unsigned int current_index = b * (CONFIG_T::V * CONFIG_T::F) + v * CONFIG_T::F + f;
                input_T current_val = x[current_index];

                current_mean += current_val;

                if (current_val < current_min) {
                    current_min = current_val;
                }
                if (current_val > current_max) {
                    current_max = current_val;
                }
            }

            current_mean = current_mean / CONFIG_T::V;

            for (unsigned int v = 0; v < CONFIG_T::V; v++) {
                // Current index for result array 'res'
                unsigned int res_base_idx = b * (CONFIG_T::V * 4 * CONFIG_T::F) + v * (4 * CONFIG_T::F) + f;

                // Assign mean, min, max to result array
                res[res_base_idx] = current_mean;
                res[res_base_idx + CONFIG_T::F] = current_min;
                res[res_base_idx + 2 * CONFIG_T::F] = current_max;

                // Append input features to result array
                unsigned int x_idx = b * (CONFIG_T::V * CONFIG_T::F) + v * CONFIG_T::F + f;
                res[res_base_idx + 3 * CONFIG_T::F] = x[x_idx];
            }
        }
    }
}

// GravNet core

/**
 * @brief Converts a real value to an index for the exponent lookup table.
 *
 * This function takes a real value (typically a distance), calculates its absolute value,
 * scales it, and converts it to an integer index for the exponent lookup table.
 * The index is capped at the maximum table size.
 *
 * @tparam input_T The data type of the input value.
 * @tparam CONFIG_T The configuration struct containing table size and scaling parameters.
 * @param x The input real value.
 * @return The calculated index for the exponent lookup table.
 */
template <class input_T, typename CONFIG_T>
inline ap_uint<CONFIG_T::exp_table_size_nbits> gravnet_idx_from_real_val(input_T x) {
    if (x < 0)
        x = -x;

    ap_uint<CONFIG_T::exp_table_size_nbits> max_idx = CONFIG_T::exp_table_size - 1;

    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);

    if (idx > max_idx) {
        return max_idx;
    }
    return (ap_uint<CONFIG_T::exp_table_size_nbits>)idx;
}

/**
 * @brief Initializes the exponent lookup table.
 *
 * This function pre-computes values for exp(-10.0f * x) and stores them in a lookup table.
 * This is an optimization to avoid costly `exp` calculations during the main processing loop.
 *
 * @tparam exp_table_T The data type of the table elements.
 * @tparam CONFIG_T The configuration struct containing table size and scaling parameters.
 * @param table_out The array to be filled with exponent values.
 */
template <class exp_table_T, typename CONFIG_T>
void gravnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    // Set exp for small distances to one to give room for optimizations
    table_out[0] = 1.0f;
    // Set exp for large distances to zero to give room for optimizations
    table_out[CONFIG_T::exp_table_size - 1] = 0.0f;

    for (unsigned i = 1; i < CONFIG_T::exp_table_size - 1; i++) {
#pragma HLS UNROLL
        float val = (float)((ap_fixed<32, 16>)(i + 1) >> CONFIG_T::exp_table_indexing_shmt);
        float res = std::exp(-10.0f * val);
        table_out[i] = (exp_table_T)res;
    }
}

/**
 * @brief A struct to hold the distance and index of a graph node.
 *
 * This is used for sorting and finding the k-nearest neighbors.
 *
 * @tparam dist_T The data type for the distance (e.g., squared Euclidean distance).
 * @tparam idx_T The data type for the node index.
 */
template <class dist_T, class idx_T> struct Node {
    dist_T dist;
    idx_T index;

    void operator=(dist_T d) {
        dist = d;
        index = 0;
    }
};

/**
 * @brief Updates the array holding the current k-nearest neighbors for a given node.
 *
 * This function inserts a new node (with its distance and index) into a sorted array
 * of k-nearest neighbors if its distance is smaller than any of the current neighbors.
 *
 * @tparam dist_T The data type for the distance.
 * @tparam idx_T The data type for the node index.
 * @tparam CONFIG_T The configuration struct containing `n_neighbors`.
 * @param new_dist The distance of the new candidate neighbor.
 * @param new_index The index of the new candidate neighbor.
 * @param knns The array of current k-nearest neighbors to be updated.
 */
template <class dist_T, class idx_T, typename CONFIG_T>
void update_knn(dist_T new_dist, idx_T new_index, Node<dist_T, idx_T> knns[CONFIG_T::n_neighbors]) {
    Node<dist_T, idx_T> current_node;
    current_node.dist = new_dist;
    current_node.index = new_index;

    for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
        if (current_node.dist < knns[n].dist) {
            Node<dist_T, idx_T> tmp = knns[n];
            knns[n] = current_node;

            // Propagate the previous current_knn element to
            // be possibly inserted afterwards and not discarded
            // if still one of the closest neighbors
            current_node = tmp;
        }
    }
}

/**
 * @brief Implements the core logic of the GravNet layer.
 *
 * This function performs the following steps:
 * 1. Calculates squared Euclidean distances between all pairs of nodes (vertices).
 * 2. For each node, finds the k-nearest neighbors.
 * 3. Calculates weights for each neighbor using `w = exp(-10 * d^2)`.
 * 4. Aggregates features from the neighbors, weighted by the weights.
 * 5. Computes the maximum and mean of the weighted features for each node.
 * 6. Concatenates the max and mean aggregations to produce the output.
 *
 * @tparam input_T Data type for input coordinates and features.
 * @tparam output_T Data type for the output features.
 * @tparam knn_dist_T Data type for the k-NN distance calculations.
 * @tparam knn_idx_T Data type for the k-NN indices.
 * @tparam exp_T Data type for the exponent lookup table.
 * @tparam weight_T Data type for the weighted features.
 * @tparam CONFIG_T The configuration struct with layer parameters.
 * @param coords Input coordinates of shape (B, V, S), flattened.
 * @param feats Input features of shape (B, V, F), flattened.
 * @param res Output features of shape (B, V, 2*F), flattened.
 */
template <class input_T, class output_T, class knn_dist_T, class knn_idx_T, class exp_T, class weight_T, typename CONFIG_T>
void gravnet_core(input_T coords[CONFIG_T::B * CONFIG_T::V * CONFIG_T::S],
                  input_T feats[CONFIG_T::B * CONFIG_T::V * CONFIG_T::F],
                  output_T res[CONFIG_T::B * CONFIG_T::V * 2 * CONFIG_T::F]) {
#ifdef __HLS_SYN__
    bool initialized = false;
    exp_table_T exp_table[CONFIG_T::exp_table_size];
#else
    static bool initialized = false;
    static exp_T exp_table[CONFIG_T::exp_table_size];
#endif

    if (!initialized) {
        gravnet_init_exp_table<exp_T, CONFIG_T>(exp_table);
        initialized = true;
    }

    Node<knn_dist_T, knn_idx_T> knns[CONFIG_T::V * CONFIG_T::n_neighbors];
    output_T fmax[CONFIG_T::F];
    output_T fsum[CONFIG_T::F];

    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        const unsigned int base_idx_feats = b * (CONFIG_T::V * CONFIG_T::F);
        const unsigned int base_idx_coords = b * (CONFIG_T::V * CONFIG_T::S);
        const unsigned int res_base_idx = b * (CONFIG_T::V * 2 * CONFIG_T::F);

        for (unsigned int v_n = 0; v_n < CONFIG_T::V * CONFIG_T::n_neighbors; v_n++) {
            knns[v_n].dist = 30000;
            knns[v_n].index = 0;
        }

        for (unsigned int i = 0; i < CONFIG_T::V; i++) {
            for (unsigned int s = 0; s < CONFIG_T::F; s++) {
                fmax[s] = -30000;
                fsum[s] = 0;
            }

            unsigned int row_offset_coords = base_idx_coords + i * CONFIG_T::S;
            unsigned int knn_offset_i = i * CONFIG_T::n_neighbors;

            // It is sufficient to iterate only over the upper part of the matrix here
            // since the euclidean squared distance matrix will be symmetric
            for (unsigned int j = i + 1; j < CONFIG_T::V; j++) {
                unsigned int col_offset_coords = base_idx_coords + j * CONFIG_T::S;
                unsigned int knn_offset_j = j * CONFIG_T::n_neighbors;

                knn_dist_T dist_sq = 0;

                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
                    input_T diff = coords[row_offset_coords + s] - coords[col_offset_coords + s];
                    dist_sq += (knn_dist_T)(diff * diff);
                }

                update_knn<knn_dist_T, knn_idx_T, CONFIG_T>(dist_sq, j, &knns[knn_offset_i]);
                update_knn<knn_dist_T, knn_idx_T, CONFIG_T>(dist_sq, i, &knns[knn_offset_j]);
            }

            for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
                knn_idx_T neighbor_idx = knns[knn_offset_i + n].index;
                knn_dist_T d = knns[knn_offset_i + n].dist;
                ap_uint<CONFIG_T::exp_table_size_nbits> idx = gravnet_idx_from_real_val<knn_dist_T, CONFIG_T>(d);
                exp_T w = exp_table[idx];

                unsigned int neighbor_offset_feats = base_idx_feats + neighbor_idx * CONFIG_T::F;

                for (unsigned int f = 0; f < CONFIG_T::F; f++) {
                    input_T feat = feats[neighbor_offset_feats + f];
                    weight_T weighted = feat * w;

                    fsum[f] += weighted;

                    if (weighted > fmax[f]) {
                        fmax[f] = weighted;
                    }
                }
            }

            unsigned int out_row_idx = res_base_idx + i * (2 * CONFIG_T::F);

            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
                res[out_row_idx + f] = fmax[f];
                res[out_row_idx + CONFIG_T::F + f] = fsum[f] / CONFIG_T::n_neighbors;
            }
        }
    }
}
