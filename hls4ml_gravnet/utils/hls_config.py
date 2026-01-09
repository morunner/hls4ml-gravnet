import hls4ml
from hls4ml_gravnet.hls4ml_extension.global_exchange import HGlobalExchange
from hls4ml_gravnet.hls4ml_extension.global_exchange_parser import parse_global_exchange
from hls4ml_gravnet.hls4ml_extension.global_exchange_template import (
    GlobalExchangeConfigTemplate,
    GlobalExchangeFunctionTemplate,
)
from hls4ml_gravnet.hls4ml_extension.gravnet_core import HGravNetCore
from hls4ml_gravnet.hls4ml_extension.gravnet_core_parser import parse_gravnet_layer
from hls4ml_gravnet.hls4ml_extension.gravnet_core_template import GravNetCoreConfigTemplate, GravNetCoreFunctionTemplate
from hls4ml_gravnet.utils.files import PROJECT_ROOT


def hls4ml_gravnet_register_extensions(backend: str):
    hls4ml.converters.register_keras_v2_layer_handler('GravNetCore', parse_gravnet_layer)
    hls4ml.converters.register_keras_v2_layer_handler('GlobalExchange', parse_global_exchange)
    hls4ml.model.layers.register_layer('GravNetCore', HGravNetCore)
    hls4ml.model.layers.register_layer('GlobalExchange', HGlobalExchange)
    backend = hls4ml.backends.get_backend(backend)
    backend.register_template(GravNetCoreConfigTemplate)
    backend.register_template(GravNetCoreFunctionTemplate)
    backend.register_template(GlobalExchangeConfigTemplate)
    backend.register_template(GlobalExchangeFunctionTemplate)
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_gravnet_core.h')
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_gravnet_bitonic_sort.h')
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_global_exchange.h')


def set_qgravnet_hls_config(hls_config: dict):
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,8,AP_RND,AP_SAT>', 'maximum': 'ap_fixed<16,8,AP_RND,AP_SAT>'}
    hls_config['Model']['Strategy'] = 'Latency'

    for layer in hls_config['LayerName'].keys():
        if 'core' in layer:
            hls_config['LayerName'][layer]['ExponentialTable'] = {'ScaleFactor': 2, 'Resolution': 16}
            hls_config['LayerName'][layer]['Precision']['coords_diff'] = 'ap_fixed<8,3>'
            hls_config['LayerName'][layer]['Precision']['exp_table'] = 'ap_ufixed<8,1>'

    hls_config['LayerName']['global_avg_pool']['Precision']['accum'] = 'ap_fixed<22,10>'


def set_converter_opts(converter_opts: dict, backend: str = 'Vitis'):
    if backend == 'Vitis':
        pass  # Leave defaults
    elif backend == 'CoyoteAccelerator':
        converter_opts['io_type'] = 'io_parallel'
        converter_opts['clock_period'] = 4


def get_build_opts(backend: str = 'Vitis') -> dict:
    build_opts = {
        'reset': True,
        'csim': True,
        'synth': True,
        'cosim': True,
        'validation': True,
    }
    if backend == 'CoyoteAccelerator':
        build_opts['csynth'] = True
        build_opts['timing_opt'] = True
        build_opts['bitfile'] = True
    else:
        build_opts['vsynth'] = True

    return build_opts
