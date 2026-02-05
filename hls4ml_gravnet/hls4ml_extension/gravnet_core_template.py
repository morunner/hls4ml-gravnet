from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml_gravnet.hls4ml_extension.gravnet_core import HGravNetCore

gravnet_core_config_template = """
struct config{index} : nnet::gravnet_core_config {{
    static const unsigned V = {V};
    static const unsigned S = {S};
    static const unsigned F = {F};
    static const unsigned n_neighbours = {n_neighbours};
    static const unsigned exp_table_size = {exp_table_size};
    static const unsigned exp_table_indexing_shmt = {exp_table_indexing_shmt};
}};\n"""

gravnet_core_function_template = (
    'nnet::gravnet_core<{input1_t}, {input2_t}, {output_t}, {accum_t}, {coords_diff_t}, {knn_dist_t}, {knn_idx_t}, '
    '{exp_table_t}, {exp_table_idx_t}, {weighted_feature_t}, {config}>({input1}, {input2}, {output});'
)
gravnet_core_include_list = [
    'nnet_utils/nnet_gravnet_core_common.h',
]


class GravNetCoreConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HGravNetCore)
        self.template = gravnet_core_config_template

    def format(self, node):
        params = self._default_config_params(node)

        params['V'] = node.get_attr('V')
        params['S'] = node.get_attr('S')
        params['F'] = node.get_attr('F')

        params['n_neighbours'] = node.get_attr('n_neighbours')

        params['exp_table_size'] = node.get_attr('exp_table_size')
        params['exp_table_indexing_shmt'] = node.get_attr('exp_table_indexing_shmt')

        return self.template.format(**params)


class GravNetCoreFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HGravNetCore, include_header=gravnet_core_include_list)
        self.template = gravnet_core_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['input1_t'] = node.get_input_variable(node.inputs[0]).type.name
        params['input2_t'] = node.get_input_variable(node.inputs[1]).type.name
        params['input1'] = node.get_input_variable(node.inputs[0]).name
        params['input2'] = node.get_input_variable(node.inputs[1]).name

        params['coords_diff_t'] = node.get_attr('coords_diff_t').name
        params['knn_dist_t'] = node.get_attr('knn_dist_t').name
        params['knn_idx_t'] = node.get_attr('knn_idx_t').name
        params['exp_table_t'] = node.get_attr('exp_table_t').name
        params['exp_table_idx_t'] = node.get_attr('exp_table_idx_t').name
        params['weighted_feature_t'] = node.get_attr('weighted_feature_t').name
        params['accum_t'] = node.get_attr('accum_t').name

        return self.template.format(**params)
