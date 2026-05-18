from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl
from hls4ml.backends.coyote_accelerator.coyote_accelerator_overlay import CoyoteOverlay
from tqdm import tqdm

from common import HLS_ROOT, METRICS_ROOT, VERTICES, project_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Measure Coyote accelerator latency and throughput with synthetic inputs.')
    parser.add_argument('-v', '--vertices', type=int, choices=VERTICES, required=True)
    parser.add_argument('--par', dest='par', type=int, default=1)
    parser.add_argument('--tag')
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 32, 64])
    parser.add_argument('--num-runs', type=int, default=10)
    parser.add_argument('--hls-root', type=Path, default=HLS_ROOT)
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--program-hacc-fpga', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = project_name(args.vertices, 'CoyoteAccelerator', args.par, args.tag)
    project_dir = Path(args.hls_root) / project
    if not project_dir.exists():
        warnings.warn(f'Missing Coyote project {project_dir}; run `make synth VERTICES_PAR={args.vertices}:{args.par}` first. Skipping.')
        return

    overlay = CoyoteOverlay(project_dir, project)

    if args.program_hacc_fpga:
        overlay.program_hacc_fpga()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(args.samples, args.vertices, 4)).astype(np.float32)
    rows = []
    for batch_size in args.batch_sizes:
        for _ in tqdm(range(args.num_runs), desc=f'{project} batch={batch_size}'):
            _, latency, throughput = overlay.predict(x, (2, 1), batch_size)
            rows.append(
                {
                    'vertices': args.vertices,
                    'project': project,
                    'batch_size': batch_size,
                    'samples': args.samples,
                    'latency_us': latency,
                    'throughput_samples_per_s': throughput,
                }
            )

    table = (
        pl.DataFrame(rows)
        .group_by(['vertices', 'project', 'batch_size', 'samples'])
        .agg(
            pl.col('latency_us').mean().alias('latency_mean_us'),
            pl.col('latency_us').std().alias('latency_std_us'),
            pl.col('throughput_samples_per_s').mean().alias('throughput_mean_samples_per_s'),
        )
    )
    args.metrics_root.mkdir(parents=True, exist_ok=True)
    out_file = args.metrics_root / f'{project}_coyote_inference.csv'

    table = (
        table
        .with_columns(
            pl.col(pl.NUMERIC_DTYPES).round(4)
        )
        .sort("batch_size")
    )

    table.write_csv(out_file)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
