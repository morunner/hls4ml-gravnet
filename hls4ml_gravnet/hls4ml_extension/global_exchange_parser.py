from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer


def parse_global_exchange(keras_layer, input_names, input_shapes, data_reader):
    layer = parse_default_keras_layer(keras_layer, input_names)

    input_shape = input_shapes[0][1:]  # first dimension of shape is batch_size

    n_vertices = input_shape[0]
    n_out_features = 4 * input_shape[1]

    layer['n_vertices'] = n_vertices
    layer['n_out_features'] = n_out_features

    output_shape = [None, n_vertices, n_out_features]

    return layer, output_shape
