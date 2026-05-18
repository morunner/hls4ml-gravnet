from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import polars as pl

from common import METRICS_ROOT


PROJECT_RE = re.compile(r'gravnet_(?P<vertices>\d+)vertices_(?P<backend>[^_]+)_(?P<par>\d+)PAR')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect Coyote inference metrics into compact CSV and Markdown tables.')
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--output-csv', default='coyote-metrics.csv')
    parser.add_argument('--output-md', default='coyote-metrics.md')
    return parser.parse_args()


def read_many(metrics_root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(metrics_root.glob(pattern))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(path) for path in files], how='diagonal')


def parse_project(project: str) -> dict[str, Any]:
    match = PROJECT_RE.search(project)
    if not match:
        return {'vertices': None, 'backend': None, 'par': None}
    return {
        'vertices': int(match.group('vertices')),
        'backend': match.group('backend'),
        'par': int(match.group('par')),
    }


def add_project_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or 'project' not in df.columns:
        return df
    parsed = [parse_project(project) for project in df['project'].to_list()]
    parsed_df = pl.DataFrame(parsed)
    base = df.drop([col for col in ('vertices', 'backend', 'par') if col in df.columns])
    return pl.concat([base, parsed_df], how='horizontal')


def cell(value: Any) -> str:
    return '' if value is None else str(value)


def write_markdown(table: pl.DataFrame, path: Path) -> None:
    columns = [
        ('vertices', 'Vertices'),
        ('par', 'PAR'),
        ('batch_size', 'Batch size'),
        ('samples', 'Samples'),
        ('latency_mean', 'Latency mean (us)'),
        ('latency_std', 'Latency std (us)'),
        ('throughput_mean', 'Throughput mean'),
    ]
    present = [(key, label) for key, label in columns if key in table.columns]
    lines = [
        '# Coyote Metrics',
        '',
        '| ' + ' | '.join(label for _, label in present) + ' |',
        '| ' + ' | '.join('---' for _ in present) + ' |',
    ]
    previous_project: tuple[Any, Any] | None = None
    for row in table.to_dicts():
        project = (row.get('vertices'), row.get('par'))
        if previous_project is not None and project != previous_project:
            lines.append('| ' + ' | '.join('---' for _ in present) + ' |')
        lines.append('| ' + ' | '.join(cell(row.get(key)) for key, _ in present) + ' |')
        previous_project = project
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    args.metrics_root.mkdir(parents=True, exist_ok=True)

    coyote = add_project_columns(read_many(args.metrics_root, 'gravnet_*_coyote_inference.csv'))
    if not coyote.is_empty():
        columns = ['vertices', 'par', 'batch_size', 'samples', 'latency_mean', 'latency_std', 'throughput_mean']
        coyote_table = coyote.select([col for col in columns if col in coyote.columns]).sort(['vertices', 'par', 'batch_size'])
        out_csv = args.metrics_root / args.output_csv
        out_md = args.metrics_root / args.output_md
        coyote_table.write_csv(out_csv)
        write_markdown(coyote_table, out_md)
        print(f'Wrote {out_csv}')
        print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()
