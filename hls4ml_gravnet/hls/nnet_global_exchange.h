#ifndef NNET_GLOBAL_EXCHANGE_H_
#define NNET_GLOBAL_EXCHANGE_H_

namespace nnet {

struct global_exchange_config {
    static const unsigned V = 128;
    static const unsigned F = 8;
    static const unsigned V_nbits = 7;
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
#pragma HLS ARRAY_PARTITION variable = x block factor = CONFIG_T ::V
#pragma HLS ARRAY_PARTITION variable = res block factor = (4 * CONFIG_T::V)

top:
    for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS PIPELINE II = 1
        input_T first_val = x[f];

        mean_T mean = first_val;
        input_T min = first_val;
        input_T max = first_val;

    update:
        for (unsigned int v = 1; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
            input_T current_val = x[v * CONFIG_T::F + f];

            mean += current_val;
            min = (current_val < min) ? current_val : min;
            max = (current_val > max) ? current_val : max;
        }

        mean = mean >> CONFIG_T::V_nbits;

    assign:
        for (unsigned int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS UNROLL
            res[v * (4 * CONFIG_T::F) + f] = (output_T)mean;
            res[v * (4 * CONFIG_T::F) + f + CONFIG_T::F] = (output_T)min;
            res[v * (4 * CONFIG_T::F) + f + 2 * CONFIG_T::F] = (output_T)max;
            res[v * (4 * CONFIG_T::F) + f + 3 * CONFIG_T::F] = (output_T)x[v * CONFIG_T::F + f];
        }
    }
}

} // namespace nnet

#endif
