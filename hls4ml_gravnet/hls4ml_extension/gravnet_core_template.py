from qgravnet.layers import GravNetCore

import hls4ml

gravnet_core_config_template = """
    struct config{index} : nnet::gravnet_core_config {{
        static const unsigned V = {V};
        static const unsigned S = {S};
        static const unsigned F = {F};
        static const unsigned n_neighbours = {n_neighbours};
        static const unsigned exp_table_size = {exp_table_size};
        static const unsigned exp_table_size_nbits = {exp_table_size_nbits};
        static const unsigned exp_table_indexing_shmt = {exp_table_size_indexing_shmt};
    }};\n"""

gravnet_core_function_template = (
    'nnet::gravnet_core<{input1_t}, {input2_t}, {output_t}, {knn_dist_t}, {knn_idx_t}, '
    '{exp_t}, {weighted_feature_t}, {config}>({input1}, {input2}, {output});'
)
gravnet_core_include_list = ['nnet_utils/nnet_gravnet_core.h']


class GravNetCoreConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(GravNetCore)
        self.template = gravnet_core_config_template

    def format(self, node):
        params = self._default_config_params(node)
        input_shapes = node.get_attr('input_shape', None)
        assert input_shapes is not None and len(input_shapes) == 2

        coord_shape = input_shapes[0]
        feat_shape = input_shapes[1]

        assert len(coord_shape) == 2
        assert len(feat_shape) == 2

        assert coord_shape[0] == feat_shape[0], 'Coords and feats must contain the same number of vertices'

        params['V'] = coord_shape[0]
        params['S'] = coord_shape[1]
        params['F'] = feat_shape[1]

        params['n_neighbours'] = node.get_attr('n_neighbours')
        params['exp_table_size'] = 32
        params['exp_table_size_nbits'] = 5
        params['exp_table_size_indexing_shmt'] = 4

        return self.template.format(**params)


class GravNetCoreFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(GravNetCore, include_header=gravnet_core_include_list)
        self.template = gravnet_core_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name

        params['knn_dist_t'] = node.get_attr('knn_dist_t').name
        params['knn_idx_t'] = node.get_attr('knn_idx_t').name
        params['exp_t'] = node.get_attr('exp_t').name
        params['weighted_feature_t'] = node.get_attr('weighted_feature_t').name
        return self.template.format(**params)
