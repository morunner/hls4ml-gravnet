#include <cmath>
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

/**
 * @brief updates the array holding the current k neares neighbors for a given node
 *
 * @tparam dist_T new_dist
 * @tparam idx_T new_index
 * @tparam CONFIG_T configuration struct
 * @param new_dist the new distance with which to update the knn array
 * @param new_index the index corresponding to the element of the new distance
 * @param knns array holding the current k neares neighbors for a given node
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
template <class input_T, class output_T, class knn_dist_T, class knn_idx_T, class exp_T, class weight_T, typename CONFIG_T>
void gravnet_core(input_T coords[CONFIG_T::B * CONFIG_T::V * CONFIG_T::S],
                  input_T feats[CONFIG_T::B * CONFIG_T::V * CONFIG_T::F],
                  output_T res[CONFIG_T::B * CONFIG_T::V * 2 * CONFIG_T::F]) {
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

                exp_T w = std::exp(-10.0 * d.to_double());

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
