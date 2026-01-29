#ifndef NNET_GLOBAL_EXCHANGE_STREAM_H_
#define NNET_GLOBAL_EXCHANGE_STREAM_H_

#include "hls_stream.h"
#include "nnet_types.h"

namespace nnet {

template <class input_T, class data_T, class mean_T, typename CONFIG_T>
void compute_stats(hls::stream<input_T> &data_in, hls::stream<input_T> &data_forward,
                   hls::stream<nnet::array<mean_T, CONFIG_T::F>> &sum_stream,
                   hls::stream<nnet::array<data_T, CONFIG_T::F>> &min_stream,
                   hls::stream<nnet::array<data_T, CONFIG_T::F>> &max_stream) {
#pragma HLS INLINE off

    nnet::array<mean_T, CONFIG_T::F> sum;
    nnet::array<data_T, CONFIG_T::F> min_val;
    nnet::array<data_T, CONFIG_T::F> max_val;

#pragma HLS ARRAY_PARTITION variable = sum.data complete
#pragma HLS ARRAY_PARTITION variable = min_val.data complete
#pragma HLS ARRAY_PARTITION variable = max_val.data complete

ReadLoop:
    for (unsigned int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS PIPELINE II = 1

        input_T curr_packet = data_in.read();

        data_forward.write(curr_packet);

    UpdateStats:
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL
            data_T val = curr_packet[f];

            if (v == 0) {
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

    sum_stream.write(sum);
    min_stream.write(min_val);
    max_stream.write(max_val);
}

template <class input_T, class output_T, class data_T, class res_T, class mean_T, typename CONFIG_T>
void distribute(hls::stream<input_T> &data_forward, hls::stream<nnet::array<mean_T, CONFIG_T::F>> &sum_stream,
                hls::stream<nnet::array<data_T, CONFIG_T::F>> &min_stream,
                hls::stream<nnet::array<data_T, CONFIG_T::F>> &max_stream, hls::stream<output_T> &res) {
#pragma HLS INLINE off

    nnet::array<mean_T, CONFIG_T::F> sum = sum_stream.read();
    nnet::array<data_T, CONFIG_T::F> min_val = min_stream.read();
    nnet::array<data_T, CONFIG_T::F> max_val = max_stream.read();

#pragma HLS ARRAY_PARTITION variable = sum.data complete
#pragma HLS ARRAY_PARTITION variable = min_val.data complete
#pragma HLS ARRAY_PARTITION variable = max_val.data complete

WriteLoop:
    for (unsigned int v = 0; v < CONFIG_T::V; v++) {
#pragma HLS PIPELINE II = 1

        input_T in_packet = data_forward.read();
        output_T out_packet;

    ConstructPacket:
        for (unsigned int f = 0; f < CONFIG_T::F; f++) {
#pragma HLS UNROLL

            res_T mean = (res_T)(sum[f] >> CONFIG_T::V_nbits);

            out_packet[f] = mean;
            out_packet[f + CONFIG_T::F] = (res_T)min_val[f];
            out_packet[f + 2 * CONFIG_T::F] = (res_T)max_val[f];
            out_packet[f + 3 * CONFIG_T::F] = (res_T)in_packet[f];
        }
        res.write(out_packet);
    }
}

template <class input_T, class output_T, class mean_T, typename CONFIG_T>
void global_exchange(hls::stream<input_T> &data, hls::stream<output_T> &res) {
#pragma HLS DATAFLOW

    typedef typename input_T::value_type data_T;
    typedef typename output_T::value_type res_T;

    // Ensure a high enough depth for this fifo to prevent deadlocks
    hls::stream<input_T> data_forward("gex_data_forward");
#pragma HLS STREAM variable = data_forward depth = CONFIG_T::V

    hls::stream<nnet::array<mean_T, CONFIG_T::F>> sum_stream("gex_sum");
    hls::stream<nnet::array<data_T, CONFIG_T::F>> min_stream("gex_min");
    hls::stream<nnet::array<data_T, CONFIG_T::F>> max_stream("gex_max");
#pragma HLS STREAM variable = sum_stream depth = 1
#pragma HLS STREAM variable = min_stream depth = 1
#pragma HLS STREAM variable = max_stream depth = 1

    compute_stats<input_T, data_T, mean_T, CONFIG_T>(data, data_forward, sum_stream, min_stream, max_stream);

    distribute<input_T, output_T, data_T, res_T, mean_T, CONFIG_T>(data_forward, sum_stream, min_stream, max_stream, res);
}

} // namespace nnet
#endif
