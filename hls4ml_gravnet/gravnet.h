#include <cmath>
#include <limits>
#include <sys/types.h>

/**
 * @brief struct containing the index and distance of a GNN node.
 *
 * Used for extracting k-nn.
 *
 * @tparam dist_T dist (euclidean squared distance)
 * @tparam idx_T index (index of the current node in the entire matrix of nodes)
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
 * @brief GravNet global exchange function
 *
 * @tparam data_T x
 * @tparam res_T res
 * @tparam CONFIG_T configuration struct
 * @param x input data
 * @param res result
 */
template <class data_T, class res_T, typename CONFIG_T>
void global_exchange(data_T x[CONFIG_T::B * CONFIG_T::V * CONFIG_T::F],
                     res_T res[CONFIG_T::B * CONFIG_T::V * 4 * CONFIG_T::F]) {
    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
            unsigned int first_v_index = b * (CONFIG_T::V * CONFIG_T::F) + 0 * CONFIG_T::F + f;
            data_T first_val = x[first_v_index];

            // Initialize
            data_T current_mean = 0;
            data_T current_min = first_val;
            data_T current_max = first_val;

            // Calculate mean, min, max
            for (unsigned int v = 0; v < CONFIG_T::V; v++) {
                unsigned int current_index = b * (CONFIG_T::V * CONFIG_T::F) + v * CONFIG_T::F + f;
                data_T current_val = x[current_index];

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

/**
 * @brief Euclidean squared distances between matrix elements
 *
 * Currently, only the euclidean squared distances between a matrix and itself
 * is supported.
 * This allows for better optimizations and thus higher performance.
 *
 * @tparam data_T A
 * @tparam res_T res
 * @tparam CONFIG_T configuration struct
 * @param A input matrix to calculate the euclidean squared distances for
 * @param res the matrix containing the euclidean squared distances
 */
template <class data_T, class res_T, typename CONFIG_T>
void euclidean_squared(data_T A[CONFIG_T::B * CONFIG_T::V * CONFIG_T::S],
                       res_T res[CONFIG_T::B * CONFIG_T::V * CONFIG_T::V]) {
    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        const unsigned int A_batch_idx = b * (CONFIG_T::V * CONFIG_T::S);
        const unsigned int res_batch_idx = b * (CONFIG_T::V * CONFIG_T::V);

        for (unsigned int i = 0; i < CONFIG_T::V; i++) {
            unsigned int A_row_idx = A_batch_idx + i * CONFIG_T::S;

            // Set diagonal to zero
            res[res_batch_idx + i * CONFIG_T::V + i] = 0;

            // Only iterate over upper triangle
            // Elements from lower triangle will contain the same values
            for (unsigned int j = i + 1; j < CONFIG_T::V; j++) {
                unsigned int A_col_idx = A_batch_idx + j * CONFIG_T::S;

                res_T sum = 0;

                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
                    data_T i_elem = A[A_row_idx + s];
                    data_T j_elem = A[A_col_idx + s];
                    data_T diff = i_elem - j_elem;
                    sum += (diff * diff);
                }

                // index of upper triangle: res[b][i][j]
                unsigned int upper_idx = res_batch_idx + i * CONFIG_T::V + j;
                res[upper_idx] = sum;

                // index of lower triangle: res[b][j][i]
                unsigned int lower_idx = res_batch_idx + j * CONFIG_T::V + i;
                res[lower_idx] = sum;
            }
        }
    }
}

/**
 * @brief updates the array holding the current k neares neighbors for a given node
 *
 * @tparam dist_T new_dist
 * @tparam idx_T new_index
 * @tparam CONFIG_T configuration struct
 * @param new_dist the new distance with which to update the knn array
 * @param new_index the index corresponding to the element of the new distance
 * @param current_knn array holding the current k neares neighbors for a given node
 */
template <class dist_T, class idx_T, typename CONFIG_T>
void update_knn(dist_T new_dist, idx_T new_index, Node<dist_T, idx_T> current_knn[CONFIG_T::n_neighbors]) {
    Node<dist_T, idx_T> current_node;
    current_node.dist = new_dist;
    current_node.index = new_index;

    for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
        if (current_node.dist < current_knn[n].dist) {
            Node<dist_T, idx_T> tmp = current_knn[n];
            current_knn[n] = current_node;

            // Propagate the previous current_knn element to
            // be possibly inserted afterwards and not discarded
            // if still one of the closest neighbors
            current_node = tmp;
        }
    }
}

/**
 * @brief calculate k nearest neighbors from euclidean squared distances in one go
 *
 * @tparam data_T A
 * @tparam dist_T out_dist
 * @tparam idx_T out_indices
 * @tparam CONFIG_T configuration struct
 * @param A input matrix with elements for which to evaluate knn
 * @param out_dist contains all distances of the k neares neighbors for each node
 * @param out_indices contains all indices of the k neares neighbors for each node
 */
template <class data_T, class dist_T, class idx_T, typename CONFIG_T>
void euclidean_squared_knn(data_T A[CONFIG_T::B * CONFIG_T::V * CONFIG_T::S],
                           dist_T out_dist[CONFIG_T::B * CONFIG_T::V * CONFIG_T::n_neighbors],
                           idx_T out_indices[CONFIG_T::B * CONFIG_T::V * CONFIG_T::n_neighbors]) {
    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        const unsigned int A_batch_idx = b * (CONFIG_T::V * CONFIG_T::S);
        const unsigned int res_batch_idx = b * (CONFIG_T::V * CONFIG_T::n_neighbors);

        for (unsigned int i = 0; i < CONFIG_T::V; i++) {
            Node<dist_T, idx_T> current_knn[CONFIG_T::n_neighbors];
            for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
                current_knn[n].dist = 32768;
                current_knn[n].index = 0;
            }

            unsigned int A_row_idx = A_batch_idx + i * CONFIG_T::S;

            for (unsigned int j = 0; j < CONFIG_T::V; j++) {
                // No need to compare an element with itself
                if (i == j)
                    continue;

                unsigned int A_col_idx = A_batch_idx + j * CONFIG_T::S;

                dist_T sum = 0;

                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
                    data_T diff = A[A_row_idx + s] - A[A_col_idx + s];
                    sum += (diff * diff);
                }

                update_knn<dist_T, idx_T, CONFIG_T>(sum, j, current_knn);
            }

            for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
                unsigned int res_idx = res_batch_idx + i * CONFIG_T::n_neighbors + n;
                out_dist[res_idx] = current_knn[n].dist;
                out_indices[res_idx] = current_knn[n].index;
            }
        }
    }
}

/**
 * @brief GravNet Core layer
 *
 * 1. Calculate euclidean squared distances
 * 2. Get k-nearest neighbors from distances d
 * 3. Calculate weights w = exp(-10d)
 * 4. Weigh input features by multiplying weight w at corresponding node index
 * 5. Calculate max and mean for each node over k-nearest weighted input features
 *
 * @tparam input_T coords, feats
 * @tparam dist_T euclidean squared distances
 * @tparam idx_T indices of k nearest neighbors
 * @tparam exp_T exponents
 * @tparam weight_T feature weights
 * @tparam res_T result
 * @tparam CONFIG_T configuration struct
 * @param coords coordinates for each node
 * @param feats features for each node
 * @param res matrix holding the GravNet core output features
 */
template <class input_T, class dist_T, class idx_T, class exp_T, class weight_T, class res_T, typename CONFIG_T>
void gravnet_core(input_T coords[CONFIG_T::B * CONFIG_T::V * CONFIG_T::S],
                  input_T feats[CONFIG_T::B * CONFIG_T::V * CONFIG_T::F],
                  res_T res[CONFIG_T::B * CONFIG_T::V * 2 * CONFIG_T::F]) {
    Node<dist_T, idx_T> current_knn[CONFIG_T::n_neighbors];
    res_T fmax[CONFIG_T::F];
    res_T fsum[CONFIG_T::F];

    for (unsigned int b = 0; b < CONFIG_T::B; b++) {
        const unsigned int base_idx_feats = b * (CONFIG_T::V * CONFIG_T::F);
        const unsigned int base_idx_coords = b * (CONFIG_T::V * CONFIG_T::S);
        const unsigned int res_base_idx = b * (CONFIG_T::V * 2 * CONFIG_T::F);

        for (unsigned int i = 0; i < CONFIG_T::V; i++) {
            for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
                current_knn[n].dist = std::numeric_limits<dist_T>::max();
                current_knn[n].index = 0;
            }

            for (unsigned int s = 0; s < CONFIG_T::F; s++) {
                fmax[s] = std::numeric_limits<res_T>::min();
                fsum[s] = 0;
            }

            unsigned int row_offset_coords = base_idx_coords + i * CONFIG_T::S;

            for (unsigned int j = 0; j < CONFIG_T::V; j++) {
                // No need to compare an element with itself
                if (i == j)
                    continue;

                unsigned int col_offset_coords = base_idx_coords + j * CONFIG_T::S;

                dist_T dist_sq = 0;

                for (unsigned int s = 0; s < CONFIG_T::S; s++) {
                    input_T diff = coords[row_offset_coords + s] - coords[col_offset_coords + s];
                    dist_sq += (dist_T)(diff * diff);
                }

                update_knn<dist_T, idx_T, CONFIG_T>(dist_sq, j, current_knn);
            }

            for (unsigned int n = 0; n < CONFIG_T::n_neighbors; n++) {
                idx_T neighbor_idx = current_knn[n].index;
                dist_T d = current_knn[n].dist;

                exp_T w = std::exp((input_T)(-10.0 * d));

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
