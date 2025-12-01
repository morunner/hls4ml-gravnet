import hls4ml
from hls4ml_gravnet.layers.global_exchange import GlobalExchange

global_exchange_config_template = """
    struct config{index} : nnet::global_exchange_config {{
        static const unsigned V = {V};
        static const unsigned F = {F};
    }};\n"""

global_exchange_function_template = 'nnet::global_exchange<{input_t}, {output_t}, {mean_t}, {config}>({input}, {output});'
global_exchange_include_list = ['nnet_utils/nnet_global_exchange.h']


class GlobalExchangeConfigTemplate(hls4ml.backends.template.LayerConfigTemplate):
    def __init__(self):
        super().__init__(GlobalExchange)
        self.template = global_exchange_config_template

    def format(self, node):
        params = self._default_config_params(node)
        input_shape = node.get_input_variable().shape
        assert len(input_shape) == 2, 'GlobalExchange currently only supports 2D inputs'
        params['V'] = input_shape[0]
        params['F'] = input_shape[1]
        return self.template.format(**params)


class GlobalExchangeFunctionTemplate(hls4ml.backends.template.FunctionCallTemplate):
    def __init__(self):
        super().__init__(GlobalExchange, include_header=global_exchange_include_list)
        self.template = global_exchange_function_template

    def format(self, node):
        params = self._default_function_params(node)
        params['mean_t'] = node.get_attr('mean_t').name
        return self.template.format(**params)
