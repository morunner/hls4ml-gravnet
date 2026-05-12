from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import MODELS_ROOT, REPO_ROOT, VERTICES, parse_model_specs


PACKAGES = ('hls4ml', 'hls4ml-gravnet', 'numpy', 'polars', 'quantized-gravnet', 'tensorflow')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check the FPL artifact software and hardware environment.')
    parser.add_argument('--vertices', type=int, nargs='+', default=list(VERTICES), choices=VERTICES)
    parser.add_argument('--models', nargs='+', help='Explicit model list as VERTICES:PAR entries, e.g. 64:1 64:2.')
    parser.add_argument('--models-root', type=Path, default=MODELS_ROOT)
    parser.add_argument('--vitis-hls-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vitis'))
    parser.add_argument('--vivado-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vivado'))
    parser.add_argument('--part', default='xcu55c-fsvh2892-2L-e')
    return parser.parse_args()


def run(cmd: list[str], timeout: int = 20, env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout, env=env)
    except FileNotFoundError:
        return {'ok': False, 'command': cmd, 'error': 'not found'}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'command': cmd, 'error': f'timeout after {timeout}s'}
    return {
        'ok': proc.returncode == 0,
        'command': cmd,
        'returncode': proc.returncode,
        'stdout': proc.stdout.strip(),
        'stderr': proc.stderr.strip(),
    }


def command_version(executable: Path | str, args: list[str] | None = None) -> dict[str, Any]:
    cmd = [str(executable), *(args or ['--version'])]
    result = run(cmd)
    result['exists'] = shutil.which(str(executable)) is not None or Path(executable).exists()
    return result


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def model_status(vertices: list[int], models_root: Path) -> dict[str, Any]:
    status: dict[str, Any] = {}
    for vertex_count in vertices:
        cfg_path = models_root / f'model_{vertex_count}V_cfg.json'
        weights_path = models_root / f'model_{vertex_count}V.weights.h5'
        missing = [str(path) for path in (cfg_path, weights_path) if not path.is_file()]
        if missing:
            status[str(vertex_count)] = {'ok': False, 'error': f'missing: {", ".join(missing)}'}
        else:
            status[str(vertex_count)] = {'ok': True, 'config': str(cfg_path), 'weights': str(weights_path)}
    return status


def import_status(module: str) -> dict[str, Any]:
    env = {**os.environ, 'TF_CPP_MIN_LOG_LEVEL': '3'}
    result = run([sys.executable, '-c', f'import {module}'], timeout=60, env=env)
    return {'ok': result['ok'], 'error': result.get('stderr') or result.get('error', '')}


def first_line(text: str | None) -> str:
    if not text:
        return ''
    return text.splitlines()[0] if text.splitlines() else text


def print_status(label: str, ok: bool, detail: str = '') -> None:
    status = 'OK' if ok else 'FAIL'
    suffix = f' - {detail}' if detail else ''
    print(f'[{status}] {label}{suffix}')


def main() -> None:
    args = parse_args()
    vertices = list(dict.fromkeys(vertices for vertices, _ in parse_model_specs(args.models))) if args.models else args.vertices
    vitis_hls = args.vitis_hls_path / 'bin' / 'vitis-run'
    vivado = args.vivado_path / 'bin' / 'vivado'
    report: dict[str, Any] = {
        'repo_root': str(REPO_ROOT),
        'git': run(['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD']),
        'python': {'executable': sys.executable, 'version': sys.version},
        'platform': platform.platform(),
        'part': args.part,
        'packages': package_versions(),
        'tools': {
            'uv': command_version('uv'),
            'vitis_hls': command_version(vitis_hls, ['-v']),
            'vivado': command_version(vivado, ['-version']),
        },
        'models': model_status(vertices, args.models_root),
        'imports': {
            'coyote_overlay': import_status('hls4ml.backends.coyote_accelerator.coyote_accelerator_overlay'),
        },
    }
    failures = []
    print(f'Artifact environment check for part {args.part}')
    print(f'Repository: {REPO_ROOT}')
    print_status('git commit', report['git']['ok'], first_line(report['git'].get('stdout')))
    print_status('Python', True, sys.version.split()[0])

    for name, version in report['packages'].items():
        if version is None:
            failures.append(f'missing Python package: {name}')
        print_status(f'Python package {name}', version is not None, version or 'not installed')

    for vertex_count, status in report['models'].items():
        if not status['ok']:
            failures.append(f'missing model files for {vertex_count} vertices')
        print_status(f'{vertex_count}V model files', status['ok'], status.get('weights') or status.get('error', ''))

    for tool_name, status in report['tools'].items():
        if not status.get('ok'):
            failures.append(f'{tool_name} version check failed')
        detail = first_line(status.get('stdout')) or first_line(status.get('stderr')) or status.get('error', '')
        print_status(tool_name, status.get('ok', False), detail)

    if not report['imports']['coyote_overlay']['ok']:
        failures.append('Coyote overlay import failed')
    print_status('Coyote overlay module', report['imports']['coyote_overlay']['ok'])

    if failures:
        print('Environment check found issues:')
        for failure in failures:
            print(f'- {failure}')
        raise SystemExit(1)
    print('Environment check passed.')


if __name__ == '__main__':
    main()
