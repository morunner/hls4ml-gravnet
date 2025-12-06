from qkeras import quantized_bits, quantized_relu, quantized_sigmoid, quantized_tanh

quantizer = quantized_bits(8, 0, 1, alpha=1.0)

keras_model_cfg = {
    'n_blocks': 2,
    'n_neighbours': 40,
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
        'output_kernel_quantizer': quantizer,
        'output_bias_quantizer': quantizer,
        'output_activation': quantized_tanh(8),
        'post_gn_activation': quantized_tanh(8),
        'post_gn_out_activation': quantized_tanh(8),
        'post_gn_gex_activation': quantized_tanh(8),
        'post_gn_relu': quantized_relu(8, 0),
        'classification_activation': quantized_sigmoid(8),
    },
}


def set_qgravnet_hls_config(hls_config: dict):
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,8,AP_RND,AP_SAT>', 'maximum': 'ap_fixed<16,8,AP_RND,AP_SAT>'}

    # Exponential table settings for gravnet core
    for layer in hls_config['LayerName'].keys():
        if 'core' in layer:
            hls_config['LayerName'][layer]['ExponentialTable'] = {'ScaleFactor': 8, 'Resolution': 128}
