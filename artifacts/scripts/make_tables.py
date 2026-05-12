from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from common import METRICS_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Regenerate compact paper tables from artifact metric CSVs.')
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    return parser.parse_args()


def read_many(metrics_root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(metrics_root.glob(pattern))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(path) for path in files], how='diagonal')


def main() -> None:
    args = parse_args()
    args.metrics_root.mkdir(parents=True, exist_ok=True)

    synth = read_many(args.metrics_root, 'gravnet_*_metrics.csv')
    if synth.is_empty():
        synth = read_many(args.metrics_root, 'all_metrics.csv')
    if not synth.is_empty():
        columns = [
            'vertices',
            'backend',
            'lut_percent',
            'dsp_percent',
            'ff_percent',
            'bram_percent',
            'cosim_latency_min',
            'cosim_ii_min',
        ]
        synth_table = synth.select([col for col in columns if col in synth.columns]).sort(['vertices', 'backend'])
        synth_table.write_csv(args.metrics_root / 'paper_table.csv')
        print(f'Wrote {args.metrics_root / "paper_table.csv"}')

    coyote = read_many(args.metrics_root, 'gravnet_*_coyote_inference.csv')
    if not coyote.is_empty():
        columns = ['vertices', 'batch_size', 'samples', 'latency_mean', 'latency_std', 'throughput_mean']
        coyote_table = coyote.select([col for col in columns if col in coyote.columns]).sort(['vertices', 'batch_size'])
        coyote_table.write_csv(args.metrics_root / 'paper_table_coyote_inference.csv')
        print(f'Wrote {args.metrics_root / "paper_table_coyote_inference.csv"}')


if __name__ == '__main__':
    main()
