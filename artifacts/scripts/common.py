from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / 'artifacts' / 'models'
HLS_ROOT = REPO_ROOT / 'artifacts' / 'hls4ml_out'
METRICS_ROOT = REPO_ROOT / 'artifacts' / 'metrics'

VERTICES = (64, 128, 256, 512)
BACKENDS = ('Vitis', 'CoyoteAccelerator')
VERTICES_PAR = ((64, 1), (64, 2), (128, 1), (128, 2), (256, 1), (512, 1))


def parse_model_specs(specs: list[str]) -> list[tuple[int, int]]:
    models = []
    for spec in specs:
        try:
            vertices_str, par_str = spec.split(':', maxsplit=1)
            vertices = int(vertices_str)
            par = int(par_str)
        except ValueError as exc:
            raise ValueError(f'Invalid model spec {spec!r}; expected VERTICES:PAR, e.g. 64:2') from exc
        if vertices not in VERTICES:
            raise ValueError(f'Unsupported vertices value {vertices}; expected one of {VERTICES}')
        if par < 1:
            raise ValueError(f'Invalid PAR value {par}; expected >= 1')
        models.append((vertices, par))
    return models


def load_model_files(vertices: int, models_root: Path = MODELS_ROOT) -> tuple[dict[str, Any], Path]:
    cfg_path = models_root / f'model_{vertices}V_cfg.json'
    weights_path = models_root / f'model_{vertices}V.weights.h5'
    if not cfg_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f'Missing model_{vertices}V_cfg.json/model_{vertices}V.weights.h5 in {models_root}')

    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg, weights_path


def project_name(vertices: int, backend: str, par: int = 1, tag: str | None = None) -> str:
    backend_name = backend.replace('Accelerator', '')
    parts = [f'gravnet_{vertices}vertices_{backend_name}_{par}PAR']
    if tag:
        parts.append(tag)
    return '_'.join(parts)


def configure_gravnet_hls(hls_config: dict[str, Any]) -> None:
    hls_config['Model']['Precision'] = {'default': 'ap_fixed<16,6>', 'maximum': 'ap_fixed<18,8>'}
    hls_config['Model']['Strategy'] = 'Latency'

    for layer, cfg in hls_config['LayerName'].items():
        precision = cfg.get('Precision', {})
        if 'core' in layer:
            cfg['ExponentialTable'] = {'ScaleFactor': 2, 'Resolution': 64}
            precision['coords_diff'] = 'ap_fixed<8, 3>'
            precision['exp_table'] = 'ap_ufixed<8, 1>'
        if 'gex' in layer:
            precision['mean'] = 'ap_fixed<22,12>'
        if 'weight' in precision:
            precision['weight'] = 'ap_fixed<8,1,AP_RND,AP_SAT>'
        if 'bias' in precision:
            precision['bias'] = 'ap_fixed<8,1,AP_RND,AP_SAT>'
        if 'accum' in precision:
            precision['accum'] = 'ap_fixed<20,8>'

    if 'global_avg_pool' in hls_config['LayerName']:
        hls_config['LayerName']['global_avg_pool']['Precision']['accum'] = 'ap_fixed<22,16>'


def build_options(backend: str) -> dict[str, Any]:
    opts: dict[str, Any] = {'reset': True, 'csim': True, 'synth': True, 'cosim': True, 'validation': True}
    if backend == 'CoyoteAccelerator':
        opts.update({'hls_clock_period': 5, 'csynth': True, 'timing_opt': True, 'bitfile': True})
    else:
        opts['vsynth'] = True
    return opts
