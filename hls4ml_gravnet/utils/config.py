from qkeras import quantized_bits, quantized_relu, quantized_sigmoid, quantized_tanh

quantizer = quantized_bits(8, 0, 1, alpha=1.0)
tanh_quantizer = quantized_tanh(8)

keras_model_cfg = {
    'n_blocks': 2,
    'n_neighbours': 8,
    'n_filters': 8,
    'n_propagate': 8,
    'n_postgn_dense_blocks': 2,
    'dense_layer_dims': {
        'input_dense': 8,
        'post_gn': 16,
        'postgn_block': 16,
        'out0': 8,
        'out1': 8,
    },
    'dense_kernel_quantizer': quantizer,
    'dense_bias_quantizer': quantizer,
    'gravnet_kwargs': {
        'fix_coordinate_space': False,
        'coordinate_kernel_quantizer': quantizer,
        'coordinate_bias_quantizer': quantizer,
        'feature_kernel_quantizer': quantizer,
        'feature_bias_quantizer': quantizer,
        'output_kernel_quantizer': quantizer,
        'output_bias_quantizer': quantizer,
        'output_activation': tanh_quantizer,
        'post_gn_activation': tanh_quantizer,
        'post_gn_out_activation': tanh_quantizer,
        'post_gn_gex_activation': tanh_quantizer,
        'post_gn_relu': quantized_relu(8, 3),
        'classification_activation': quantized_sigmoid(8),
        'other_kernel_initializer': 'glorot_uniform',
    },
}
