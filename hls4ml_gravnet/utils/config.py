from qkeras import quantized_bits, quantized_sigmoid

quantizer = quantized_bits(8, 0, 1, alpha=1.0)
act_quantizer = 'quantized_relu(12,4)'

keras_model_cfg = {
    'n_blocks': 2,
    'n_neighbours': 8,
    'n_filters': 8,
    'n_propagate': 8,
    'n_postgn_dense_blocks': 2,
    'distance_metric': 'l1',
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
        'output_activation': act_quantizer,
        'post_gn_activation': act_quantizer,
        'post_gn_out_activation': act_quantizer,
        'post_gn_gex_activation': act_quantizer,
        'post_gn_relu': act_quantizer,
        'classification_activation': quantized_sigmoid(8),
        'regression_kernel_quantizer': quantized_bits(16, 8, 1, alpha=1.0),
        'regression_bias_quantizer': quantized_bits(16, 8, 1, alpha=1.0),
        'other_kernel_initializer': 'glorot_uniform',
    },
}
