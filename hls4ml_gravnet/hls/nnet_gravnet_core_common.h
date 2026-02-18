#ifndef NNET_GRAVNET_COMMON_H_
#define NNET_GRAVNET_COMMON_H_

#include "ap_int.h"
#include <cmath>

namespace nnet {

template <class coord_T, class res_T, class diff_T> struct l2_squared {
    static res_T dist(coord_T a, coord_T b) {
#pragma HLS INLINE
        diff_T diff = a - b;
        return (res_T)(diff * diff);
    }
};

template <class coord_T, class res_T, class diff_T> struct l1 {
    static res_T dist(coord_T a, coord_T b) {
#pragma HLS INLINE
        diff_T tmp = a - b;
        res_T diff = (tmp > 0) ? (res_T)tmp : (res_T)(-tmp);
        return diff;
    }
};

struct gravnet_core_config {
    static const unsigned V = 128;
    static const unsigned S = 4;
    static const unsigned F = 8;
    static const unsigned n_neighbours = 32;
    static const unsigned exp_table_size = 32;
    static const unsigned exp_table_indexing_shmt = 4;
    template <class coord_T, class res_T, class diff_T> using distance_fn = nnet::l2_squared<coord_T, res_T, diff_T>;
};

template <typename T> struct gravnet_core_limits;

template <int W, int I, ap_q_mode Q, ap_o_mode O, int N> struct gravnet_core_limits<ap_fixed<W, I, Q, O, N>> {
    typedef ap_fixed<W, I, Q, O, N> T;

    static constexpr double min() { return -(1 << (I - 1)); }

    static constexpr double max() { return (1 << (I - 1)) - (1.0 / (1 << (W - I))); }

    static T min_val() { return (T)min(); }
    static T max_val() { return (T)max(); }
};

template <class input_T, class exp_table_idx_T, typename CONFIG_T> unsigned int gravnet_idx_from_real_val(input_T x) {
#pragma HLS INLINE
    if (x < 0)
        x = -x;

    exp_table_idx_T max_idx = CONFIG_T::exp_table_size - 1;

    ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt> idx =
        ((ap_fixed<x.width + CONFIG_T::exp_table_indexing_shmt, x.iwidth + CONFIG_T::exp_table_indexing_shmt>)x
         << CONFIG_T::exp_table_indexing_shmt);

    if (idx > max_idx)
        return (unsigned int)max_idx;

    return (unsigned int)idx;
}

template <class exp_table_T, typename CONFIG_T>
void gravnet_init_exp_table(exp_table_T table_out[CONFIG_T::exp_table_size]) {
    table_out[0] = 1.0f;
    table_out[CONFIG_T::exp_table_size - 1] = 0.0f;

    for (unsigned i = 1; i < CONFIG_T::exp_table_size - 1; i++) {
#pragma HLS UNROLL
        float val = (float)((ap_fixed<32, 16>)(i) >> CONFIG_T::exp_table_indexing_shmt);
        float res = std::exp(-10.0f * val);
        table_out[i] = (exp_table_T)res;
    }
}

} // namespace nnet

#endif
