#ifndef NNET_GLOBAL_EXCHANGE_H_
#define NNET_GLOBAL_EXCHANGE_H_

namespace nnet {

struct global_exchange_config {
    static const unsigned V = 128;
    static const unsigned F = 8;
};

/**
 * @brief Performs global exchange by calculating mean, min, and max for each feature across all vertices.
 * Input Shape: (V, F)
 * Output Shape: (V, 4F) [mean, min, max, x]
 *
 * @tparam input_T The data type of the input features.
 * @tparam output_T The data type of the output features.
 * @tparam mean_T The data type for mean calculation.
 * @tparam CONFIG_T The configuration struct containing dimensions V (vertices) and F (features).
 * @param x Input data array of shape (V, F), flattened.
 * @param res Output data array of shape (V, 4*F), flattened.
 */
template <class input_T, class output_T, class mean_T, typename CONFIG_T>
void global_exchange(input_T x[CONFIG_T::V * CONFIG_T::F], output_T res[CONFIG_T::V * 4 * CONFIG_T::F]) {
    for (unsigned int f = 0; f < CONFIG_T::F; f++) {
        input_T first_val = x[f];

        mean_T current_mean = 0;
        input_T current_min = first_val;
        input_T current_max = first_val;

        for (unsigned int v = 0; v < CONFIG_T::V; v++) {
            unsigned int current_index = v * CONFIG_T::F + f;
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
            unsigned int res_base_idx = v * (4 * CONFIG_T::F) + f;

            res[res_base_idx] = (output_T)current_mean;
            res[res_base_idx + CONFIG_T::F] = (output_T)current_min;
            res[res_base_idx + 2 * CONFIG_T::F] = (output_T)current_max;

            unsigned int x_idx = v * CONFIG_T::F + f;
            res[res_base_idx + 3 * CONFIG_T::F] = (output_T)x[x_idx];
        }
    }
}

} // namespace nnet

#endif
