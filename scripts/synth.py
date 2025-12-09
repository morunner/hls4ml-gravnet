import argparse
import os

from qgravnet.factory import QGravNetFactory
from sklearn.metrics import roc_auc_score

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
from utils.config import set_qgravnet_hls_config
from utils.data import load_processed
from utils.evaluation import load_run, response_rmse
from utils.files import HLS4ML_OUT_PATH, PROJECT_ROOT, RESULTS_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='SynthesizeGravNet', description='Synthesize GravNet with hls4ml')
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-n', '--project_name')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']

    model_cfg, weights_path, _, datapath = load_run(RESULTS_PATH / 'train_new_quantization_cfg')

    D = load_processed(datapath)

    keras_model = QGravNetFactory(**model_cfg).create_keras_model(n_vertices=128, n_features=4)
    keras_model.load_weights(weights_path)
    keras_model.compile()

    # Validate predictions
    test_energy_pred, test_pid_pred = keras_model.predict(D['X_hits_test'])
    test_energy_pred *= 100
    test_response_rmse = response_rmse(D['y_energy_test'], test_energy_pred)
    test_auc = roc_auc_score(D['y_pid_test'], test_pid_pred)
    print(f'Response RMSE: {test_response_rmse:.3f}, AUC: {test_auc:.3f}')

    hls4ml.converters.register_keras_v2_layer_handler('GravNetCore', parse_gravnet_layer)
    hls4ml.converters.register_keras_v2_layer_handler('GlobalExchange', parse_global_exchange)
    hls4ml.model.layers.register_layer('GravNetCore', HGravNetCore)
    hls4ml.model.layers.register_layer('GlobalExchange', HGlobalExchange)
    backend = hls4ml.backends.get_backend('Vitis')
    backend.register_template(GravNetCoreConfigTemplate)
    backend.register_template(GravNetCoreFunctionTemplate)
    backend.register_template(GlobalExchangeConfigTemplate)
    backend.register_template(GlobalExchangeFunctionTemplate)
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_gravnet_core.h')
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_gravnet_bitonic_sort.h')
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_global_exchange.h')

    hls_config = hls4ml.utils.config_from_keras_model(model=keras_model, granularity='name', backend='Vitis')
    set_qgravnet_hls_config(hls_config)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model=keras_model,
        hls_config=hls_config,
        output_dir=str(HLS4ML_OUT_PATH / args.project_name),
        project_name=args.project_name,
        backend='Vitis',
    )
    hls_model.compile()
    hls_model.build(
        csim=True,
        synth=True,
        cosim=True,
        validation=True,
    )


if __name__ == '__main__':
    main()
