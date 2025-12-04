# Synthesize a small model containing the GravNetCore layer

import argparse
import os

from hls4ml_extension.gravnet_core_parser import parse_gravnet_layer
from train import model_cfg  # use the same params as when training the 'real' QGravNet

import hls4ml
from hls4ml_gravnet.hls4ml_extension.gravnet_core import HGravNetCore
from hls4ml_gravnet.hls4ml_extension.gravnet_core_template import GravNetCoreConfigTemplate, GravNetCoreFunctionTemplate
from hls4ml_gravnet.keras_models.qgravnet_minimal import QGravNetMinimalFactory
from utils.files import HLS4ML_OUT_PATH, PROJECT_ROOT
from utils.training import generate_train_data_random

N_VERTICES = 128
N_FEATURES = 4
N_SAMPLES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='SynthGravNetMinimal', description='Synthesize a minimal GravNet Core model with hls4ml'
    )
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-n', '--project_name')

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']

    x_rand, y_rand = generate_train_data_random(N_VERTICES, N_FEATURES, N_SAMPLES)

    keras_model = QGravNetMinimalFactory(n_neighbours=model_cfg['n_neighbours']).create_keras_model(
        n_vertices=N_VERTICES, n_features=N_FEATURES
    )
    keras_model.compile(optimizer='adam', loss={'regression': 'mse', 'classification': 'binary_crossentropy'})

    keras_model.fit(x_rand, [y_rand[:, 0], y_rand[:, 1]])

    hls4ml.converters.register_keras_v2_layer_handler('GravNetCore', parse_gravnet_layer)
    hls4ml.model.layers.register_layer('GravNetCore', HGravNetCore)
    backend = hls4ml.backends.get_backend('Vitis')
    backend.register_template(GravNetCoreConfigTemplate)
    backend.register_template(GravNetCoreFunctionTemplate)
    backend.register_source(PROJECT_ROOT / 'hls4ml_gravnet' / 'hls' / 'nnet_gravnet_core.h')

    hls_config = hls4ml.utils.config_from_keras_model(model=keras_model, granularity='model', backend='Vitis')
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
