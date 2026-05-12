from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from common import BACKENDS, HLS_ROOT, MODELS_ROOT, VERTICES, parse_model_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Launch synthesis for all paper GravNet models.')
    parser.add_argument('--vertices', type=int, nargs='+', default=list(VERTICES), choices=VERTICES)
    parser.add_argument('--backends', nargs='+', default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument('--par', type=int, default=1)
    parser.add_argument(
        '--models',
        nargs='+',
        help='Explicit model list as VERTICES:PAR entries, e.g. 64:1 64:2 128:1.',
    )
    parser.add_argument('--tag', default=None, help='Optional suffix for generated project names.')
    parser.add_argument('--models-root', type=Path, default=MODELS_ROOT)
    parser.add_argument('--hls-root', type=Path, default=HLS_ROOT)
    parser.add_argument('--vitis-hls-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vitis'))
    parser.add_argument('--vivado-path', type=Path, default=Path('/tools/Xilinx/2025.1/Vivado'))
    parser.add_argument('--part', default='xcu55c-fsvh2892-2L-e')
    parser.add_argument('--test-vectors', type=int, default=16)
    parser.add_argument('--jobs', type=int, default=0, help='Maximum concurrent syntheses. Use 0 to launch all.')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def build_commands(args: argparse.Namespace, script: Path) -> list[list[str]]:
    commands = []
    models = parse_model_specs(args.models) if args.models else [(vertices, args.par) for vertices in args.vertices]
    for vertices, par in models:
        for backend in args.backends:
            cmd = [
                'uv',
                'run',
                'python',
                str(script),
                '--vertices',
                str(vertices),
                '--backend',
                backend,
                '--par',
                str(par),
                '--models-root',
                str(args.models_root),
                '--hls-root',
                str(args.hls_root),
                '--vitis-hls-path',
                str(args.vitis_hls_path),
                '--vivado-path',
                str(args.vivado_path),
                '--part',
                args.part,
                '--test-vectors',
                str(args.test_vectors),
            ]
            if args.tag:
                cmd.extend(['--tag', args.tag])
            commands.append(cmd)
    return commands


def run_commands(commands: list[list[str]], jobs: int) -> None:
    max_jobs = len(commands) if jobs == 0 else jobs
    if max_jobs < 1:
        raise ValueError('--jobs must be >= 0')

    running: list[tuple[list[str], subprocess.Popen[bytes]]] = []
    failures: list[tuple[list[str], int]] = []
    pending = iter(commands)

    def launch_until_full() -> None:
        while len(running) < max_jobs:
            try:
                cmd = next(pending)
            except StopIteration:
                break
            print('Launching:', ' '.join(cmd), flush=True)
            running.append((cmd, subprocess.Popen(cmd)))

    launch_until_full()
    while running:
        completed = False
        for cmd, proc in list(running):
            returncode = proc.poll()
            if returncode is None:
                continue
            running.remove((cmd, proc))
            completed = True
            if returncode != 0:
                failures.append((cmd, returncode))
            launch_until_full()
        if not completed:
            time.sleep(1)

    if failures:
        for cmd, returncode in failures:
            print(f'Command failed with exit code {returncode}: {" ".join(cmd)}')
        raise subprocess.CalledProcessError(failures[0][1], failures[0][0])


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name('synth.py')
    commands = build_commands(args, script)

    for cmd in commands:
        print(' '.join(cmd))
    if not args.dry_run:
        run_commands(commands, args.jobs)


if __name__ == '__main__':
    main()
