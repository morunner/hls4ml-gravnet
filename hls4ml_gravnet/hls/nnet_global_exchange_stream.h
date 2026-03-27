#ifndef NNET_GLOBAL_EXCHANGE_STREAM_H_
#define NNET_GLOBAL_EXCHANGE_STREAM_H_

#include "ap_fixed.h"
#include "hls_stream.h"
#include "nnet_types.h"

namespace nnet {

struct global_exchange_config {
    static const unsigned V = 128;
    static const unsigned F = 8;
    static const unsigned V_nbits = 7;
};

template <class input_T, class data_T, class mean_T, typename CONFIG_T>
void compute_stats(hls::stream<input_T> &data_in, hls::stream<input_T> &data_forward,
                   hls::stream<nnet::array<mean_T, CONFIG_T::F>> &sum_stream,
                   hls::stream<nnet::array<data_T, CONFIG_T::F>> &min_stream,
                   hls::stream<nnet::array<data_T, CONFIG_T::F>> &max_stream) {
#pragma HLS INLINE off

    constexpr unsigned n_pack = input_T::size / CONFIG_T::F;
    constexpr unsigned n_iterations = CONFIG_T::V / n_pack;

    nnet::array<mean_T, CONFIG_T::F> sum;
    nnet::array<data_T, CONFIG_T::F> min_val;
    nnet::array<data_T, CONFIG_T::F> max_val;

#pragma HLS ARRAY_PARTITION variable = sum.data complete
#pragma HLS ARRAY_PARTITION variable = min_val.data complete
#pragma HLS ARRAY_PARTITION variable = max_val.data complete

    for (unsigned int v = 0; v < n_iterations; v++) {
#pragma HLS PIPELINE II = 1

        input_T curr_packet = data_in.read();
        data_forward.write(curr_packet);

        for (unsigned int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL

            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
                data_T val = curr_packet[p * CONFIG_T::F + f];

                if (v == 0 && p == 0) {
                    sum[f] = (mean_T)val;
                    min_val[f] = val;
                    max_val[f] = val;
                } else {
                    sum[f] += (mean_T)val;
                    if (val < min_val[f])
                        min_val[f] = val;
                    if (val > max_val[f])
                        max_val[f] = val;
                }
            }
        }
    }

    sum_stream.write(sum);
    min_stream.write(min_val);
    max_stream.write(max_val);
}

template <class input_T, class output_T, class data_T, class res_T, class mean_T, typename CONFIG_T>
void distribute(hls::stream<input_T> &data_forward, hls::stream<nnet::array<mean_T, CONFIG_T::F>> &sum_stream,
                hls::stream<nnet::array<data_T, CONFIG_T::F>> &min_stream,
                hls::stream<nnet::array<data_T, CONFIG_T::F>> &max_stream, hls::stream<output_T> &res) {
#pragma HLS INLINE off

    constexpr unsigned n_pack = input_T::size / CONFIG_T::F;
    constexpr unsigned n_iterations = CONFIG_T::V / n_pack;

    constexpr unsigned out_F = 4 * CONFIG_T::F;

    nnet::array<mean_T, CONFIG_T::F> sum = sum_stream.read();
    nnet::array<data_T, CONFIG_T::F> min_val = min_stream.read();
    nnet::array<data_T, CONFIG_T::F> max_val = max_stream.read();

#pragma HLS ARRAY_PARTITION variable = sum.data complete
#pragma HLS ARRAY_PARTITION variable = min_val.data complete
#pragma HLS ARRAY_PARTITION variable = max_val.data complete

WriteLoop:
    for (unsigned int v = 0; v < n_iterations; v++) {
#pragma HLS PIPELINE II = 1

        input_T in_packet = data_forward.read();
        output_T out_packet;

    ConstructPack:
        for (unsigned int p = 0; p < n_pack; p++) {
#pragma HLS UNROLL

        ConstructFeatures:
            for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL

                res_T mean = (res_T)(sum[f] >> CONFIG_T::V_nbits);

                unsigned int base = p * out_F;

                out_packet[base + f] = mean;
                out_packet[base + f + CONFIG_T::F] = (res_T)min_val[f];
                out_packet[base + f + 2 * CONFIG_T::F] = (res_T)max_val[f];
                out_packet[base + f + 3 * CONFIG_T::F] = (res_T)in_packet[p * CONFIG_T::F + f];
            }
        }
        res.write(out_packet);
    }
}

template <class input_T, class output_T, class mean_T, typename CONFIG_T>
void global_exchange(hls::stream<input_T> &data, hls::stream<output_T> &res) {
#pragma HLS DATAFLOW

    typedef typename input_T::value_type data_T;
    typedef typename output_T::value_type res_T;

    constexpr unsigned n_pack = input_T::size / CONFIG_T::F;

    // Ensure a high enough depth for this fifo to prevent deadlocks
    hls::stream<input_T> data_forward;
#pragma HLS STREAM variable = data_forward depth = CONFIG_T::V / n_pack

    hls::stream<nnet::array<mean_T, CONFIG_T::F>> sum_stream;
    hls::stream<nnet::array<data_T, CONFIG_T::F>> min_stream;
    hls::stream<nnet::array<data_T, CONFIG_T::F>> max_stream;
#pragma HLS STREAM variable = sum_stream depth = 1
#pragma HLS STREAM variable = min_stream depth = 1
#pragma HLS STREAM variable = max_stream depth = 1

    compute_stats<input_T, data_T, mean_T, CONFIG_T>(data, data_forward, sum_stream, min_stream, max_stream);

    distribute<input_T, output_T, data_T, res_T, mean_T, CONFIG_T>(data_forward, sum_stream, min_stream, max_stream, res);
}

} // namespace nnet
#endif
