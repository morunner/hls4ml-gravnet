import argparse
import os

from qgravnet.factory import QGravNetFactory
from sklearn.metrics import roc_auc_score
from utils.data import load_processed
from utils.evaluation import load_run, response_rmse
from utils.files import HLS4ML_OUT_PATH, RESULTS_PATH
from utils.hls_config import get_build_opts, hls4ml_gravnet_register_extensions, set_converter_opts, set_qgravnet_hls_config

import hls4ml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='SynthesizeGravNet', description='Synthesize GravNet with hls4ml')
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-viv', '--vivado_path', default='')  # Optional
    parser.add_argument('-n', '--project_name')
    parser.add_argument('-d', '--description', default='')
    parser.add_argument('-r', '--reuse', type=int)
    parser.add_argument('-b', '--backend', default='Vitis')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']
    os.environ['PATH'] = args.vivado_path + 'bin:' + os.environ['PATH']

    model_cfg, weights_path, _, datapath = load_run(RESULTS_PATH / args.project_name)

    D = load_processed(datapath)

    keras_model = QGravNetFactory(**model_cfg).create_keras_model(n_vertices=64, n_features=4)
    keras_model.load_weights(weights_path)
    keras_model.compile()

    # Validate predictions
    test_energy_pred, test_pid_pred = keras_model.predict(D['X_hits_test'])
    test_response_rmse = response_rmse(D['y_energy_test'], test_energy_pred)
    test_auc = roc_auc_score(D['y_pid_test'], test_pid_pred)
    print(f'Response RMSE: {test_response_rmse:.3f}, AUC: {test_auc:.3f}')

    hls4ml_gravnet_register_extensions(args.backend)

    hls_config = hls4ml.utils.config_from_keras_model(model=keras_model, granularity='name', default_reuse_factor=args.reuse)
    set_qgravnet_hls_config(hls_config)

    proj_name = args.project_name if args.description == '' else f'{args.project_name}_{args.description}'
    converter_opts = {
        'model': keras_model,
        'hls_config': hls_config,
        'backend': args.backend,
        'output_dir': str(HLS4ML_OUT_PATH / proj_name),
        'project_name': args.project_name,
    }
    set_converter_opts(converter_opts, args.backend)
    hls_model = hls4ml.converters.convert_from_keras_model(**converter_opts)
    hls_model.compile()

    build_opts = get_build_opts(args.backend)
    hls_model.build(**build_opts)


if __name__ == '__main__':
    main()
