import keras.activations as activations
from qkeras import quantized_bits, quantized_sigmoid, quantized_tanh, quantized_relu

quantizer = quantized_bits(8, 0, 1, alpha=1.0)
act_quantizer = 'quantized_relu(12,4)'

ACT_MAP = {
    quantized_relu: activations.relu,
    quantized_sigmoid: activations.sigmoid,
    quantized_tanh: activations.tanh,
}

keras_model_cfg = {
    'n_blocks': 2,
    'n_neighbours': 8,
    'n_dimensions': 2, # of spatial dimensions in the learned coordinate space
    'n_propagate': 8, 
    'n_filters': 8,
    'n_postgn_dense_blocks': 2,
    'distance_metric': 'l1', 
    'neighbour_selector': 'binned',
    'dense_layer_dims': {
        'input_dense': 8,
        'post_gn': 16,
        'postgn_block': 16,
        'out0': 8,
        'out1': 8,
    },
    'dense_kernel_quantizer': quantizer,
    'dense_bias_quantizer': quantizer,
    'gravnet_cfg': {
        'fix_coordinate_space': False,
        'coordinate_kernel_quantizer': quantizer,
        'coordinate_bias_quantizer': quantizer,
        'coordinate_activation': quantized_tanh(8),
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
    'selector_cfg': {
        'bins_per_axis': 16,
        'window': 1,
        'clip_min': -1.0,
        'clip_max': 1.0,
    },
    # --- training-time overflow control ---
    'overflow_regularization_cfg': {
        'max_per_bin': 2,              # hardware capacity C
        'overflow_lambda': 1e-5,       # 0 disables
    },
}


def remove_quantization_from_config(obj):
    """Recursively remove quantization-specific keys and convert quantizer objects to strings in a config dict."""
    if isinstance(obj, dict):
        new = {}

        for k, v in obj.items():

            if k.endswith("_quantizer"):
                continue

            if type(v) in ACT_MAP:
                new[k] = ACT_MAP[type(v)]
                continue

            if isinstance(v, str) and v.startswith("quantized_"):
                base = v.split("(")[0]
                new[k] = getattr(activations, base.replace("quantized_", ""))
                continue

            new[k] = remove_quantization_from_config(v)

        return new

    return obj