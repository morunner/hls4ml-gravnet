from __future__ import annotations

import argparse
import os
from pathlib import Path

import hls4ml
import numpy as np
import polars as pl
from hls4ml_gravnet.hls4ml_extension.register_extensions import register_extensions
from qgravnet import QGravNetFactory

from collect_synth_metrics import collect_one
from common import (
    BACKENDS,
    HLS_ROOT,
    METRICS_ROOT,
    MODELS_ROOT,
    VERTICES,
    build_options,
    configure_gravnet_hls,
    load_model_files,
    project_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Synthesize one paper GravNet model and write synthesis metrics.')
    parser.add_argument('-v', '--vertices', type=int, choices=VERTICES, required=True)
    parser.add_argument('-b', '--backend', choices=BACKENDS, default='Vitis')
    parser.add_argument('-p', '--par', dest='par', type=int, default=1)
    parser.add_argument('-r', '--reuse', type=int, default=1)
    parser.add_argument('-n', '--project-name')
    parser.add_argument('--tag')
    parser.add_argument('--models-root', type=Path, default=MODELS_ROOT)
    parser.add_argument('--hls-root', type=Path, default=HLS_ROOT)
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--vitis-hls-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vitis'))
    parser.add_argument('--vivado-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vivado'))
    parser.add_argument('--part', default='xcu55c-fsvh2892-2L-e')
    parser.add_argument('--test-vectors', type=int, default=16)
    return parser.parse_args()


def prepend_tool_path(path: Path) -> None:
    if (path / 'bin').is_dir():
        os.environ['PATH'] = f'{path / "bin"}:{os.environ["PATH"]}'


def main() -> None:
    args = parse_args()
    prepend_tool_path(args.vitis_hls_path)
    prepend_tool_path(args.vivado_path)

    model_cfg, weights_path = load_model_files(args.vertices, args.models_root)
    keras_model = QGravNetFactory(**model_cfg).create_keras_model(args.vertices, 4)
    keras_model.load_weights(weights_path)
    keras_model.summary()

    register_extensions(backend=args.backend)
    hls_config = hls4ml.utils.config_from_keras_model(
        keras_model,
        granularity='name',
        backend=args.backend,
        default_reuse_factor=args.reuse,
    )
    configure_gravnet_hls(hls_config)

    name = args.project_name or project_name(args.vertices, args.backend, args.par, args.tag)
    out_dir = args.hls_root / name
    converter_opts = {
        'model': keras_model,
        'hls_config': hls_config,
        'backend': args.backend,
        'io_type': 'io_stream',
        'output_dir': str(out_dir),
        'project_name': name,
        'part': args.part,
        'n_pack': args.par,
    }
    if args.backend == 'CoyoteAccelerator':
        converter_opts['clock_period'] = 4

    hls_model = hls4ml.converters.convert_from_keras_model(**converter_opts)
    hls_model.compile()
    x = np.random.default_rng(0).normal(size=(args.test_vectors, args.vertices, 4)).astype(np.float32)
    hls_model.predict(x)

    print(f'Building {out_dir}')
    hls_model.build(**build_options(args.backend))

    args.metrics_root.mkdir(parents=True, exist_ok=True)
    metrics_file = args.metrics_root / f'{name}_metrics.csv'
    pl.DataFrame(
        [{**{'vertices': args.vertices, 'project': name}, **collect_one(out_dir, args.backend)}]
    ).write_csv(metrics_file)
    print(f'Wrote {metrics_file}')


if __name__ == '__main__':
    main()
