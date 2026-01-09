# Synthesize a small model containing the GravNetCore layer

import argparse
import os

import hls4ml
from hls4ml_gravnet.keras.qgravnet_minimal import QGravNetMinimalFactory
from utils.config import hls4ml_gravnet_register_extensions, set_qgravnet_hls_config
from utils.data import load_processed
from utils.evaluation import load_run, response_rmse
from utils.files import HLS4ML_OUT_PATH, RESULTS_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='SynthGravNetMinimal', description='Synthesize a minimal GravNet Core model with hls4ml'
    )
    parser.add_argument('-vhls', '--vitis_hls_path')
    parser.add_argument('-n', '--project_name')
    parser.add_argument('-r', '--reuse', type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['PATH'] = args.vitis_hls_path + 'bin:' + os.environ['PATH']

    model_cfg, _, _, datapath = load_run(RESULTS_PATH / 'nn_8')
    D = load_processed(datapath)

    keras_model = QGravNetMinimalFactory(n_neighbours=model_cfg['n_neighbours']).create_keras_model(
        n_vertices=128, n_features=4
    )
    keras_model.compile(optimizer='adam', loss={'regression': response_rmse, 'classification': 'binary_crossentropy'})

    keras_model.fit(
        x=D['X_hits_train'],
        y={
            'regression': D['y_energy_train'],
            'classification': D['y_pid_train'],
        },
    )
    print(keras_model.summary())

    hls4ml_gravnet_register_extensions()

    hls_config = hls4ml.utils.config_from_keras_model(
        model=keras_model, granularity='name', backend='Vitis', default_reuse_factor=32
    )
    set_qgravnet_hls_config(hls_config)
    hls_config['LayerName']['q_dense']['ReuseFactor'] = args.reuse
    hls_config['Model']['PipelineStyle'] = 'dataflow'

    hls_model = hls4ml.converters.convert_from_keras_model(
        model=keras_model,
        hls_config=hls_config,
        output_dir=str(HLS4ML_OUT_PATH / args.project_name),
        project_name=args.project_name,
        backend='Vitis',
    )
    hls_model.compile()
    hls_model.build(csim=True, synth=True, cosim=True, validation=True, vsynth=True)


if __name__ == '__main__':
    main()
