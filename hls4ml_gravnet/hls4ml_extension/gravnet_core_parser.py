from hls4ml.converters.keras_v2_to_hls import parse_default_keras_layer


def parse_gravnet_layer(keras_layer, input_names, input_shapes, data_reader):
    assert keras_layer['class_name'] == 'GravNetCore'
    layer = parse_default_keras_layer(keras_layer, input_names)

    if len(input_shapes) != 2:
        raise ValueError('Wrong number of inputs passed to GravNet core. Requires two inputs: coords and feats')

    coords_shape = input_shapes[0][1:]  # first dimension of shape is batch size ('None')
    feats_shape = input_shapes[1][1:]

    if coords_shape[0] != feats_shape[0]:
        raise ValueError('Coords and feats must have the same number of vertices')

    n_vertices = coords_shape[0]
    n_in_feat = feats_shape[1]
    n_out_features = 2 * n_in_feat

    layer['V'] = n_vertices
    layer['S'] = coords_shape[1]
    layer['F'] = n_in_feat
    layer['n_neighbours'] = keras_layer['config']['n_neighbours']
    layer['distance_metric'] = keras_layer['config']['distance_metric']

    output_shape = [None, n_vertices, n_out_features]
    return layer, output_shape
