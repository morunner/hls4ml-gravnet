from importlib import resources

import hls4ml
from hls4ml_gravnet.hls4ml_extension.global_exchange import HGlobalExchange
from hls4ml_gravnet.hls4ml_extension.global_exchange_parser import parse_global_exchange
from hls4ml_gravnet.hls4ml_extension.global_exchange_template import (
    GlobalExchangeConfigTemplate,
    GlobalExchangeFunctionTemplate,
)
from hls4ml_gravnet.hls4ml_extension.gravnet_core import HGravNetCore
from hls4ml_gravnet.hls4ml_extension.gravnet_core_parser import parse_gravnet_layer
from hls4ml_gravnet.hls4ml_extension.gravnet_core_template import (
    GravNetCoreConfigTemplate,
    GravNetCoreFunctionTemplate,
)


def register_extensions(backend: str):
    hls4ml.converters.register_keras_v2_layer_handler('GravNetCore', parse_gravnet_layer)
    hls4ml.converters.register_keras_v2_layer_handler('GlobalExchange', parse_global_exchange)
    hls4ml.model.layers.register_layer('GravNetCore', HGravNetCore)
    hls4ml.model.layers.register_layer('GlobalExchange', HGlobalExchange)
    backend = hls4ml.backends.get_backend(backend)
    backend.register_template(GravNetCoreConfigTemplate)
    backend.register_template(GravNetCoreFunctionTemplate)
    backend.register_template(GlobalExchangeConfigTemplate)
    backend.register_template(GlobalExchangeFunctionTemplate)

    hls_files = resources.files('hls4ml_gravnet.hls')
    filenames = [
        'nnet_gravnet_core_common.h',
        'nnet_gravnet_core.h',
        'nnet_gravnet_core_stream.h',
        'nnet_gravnet_bitonic_sort.h',
        'nnet_gravnet_bitonic_sort_stream.h',
        'nnet_global_exchange.h',
        'nnet_global_exchange_stream.h',
    ]
    for fname in filenames:
        with resources.as_file(hls_files / fname) as source_path:
            backend.register_source(str(source_path))
