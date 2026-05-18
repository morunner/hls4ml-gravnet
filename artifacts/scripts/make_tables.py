from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any

import polars as pl

from common import METRICS_ROOT, REPO_ROOT


PROJECT_RE = re.compile(r'gravnet_(?P<vertices>\d+)vertices_(?P<backend>[^_]+)_(?P<par>\d+)PAR')
COYOTE_REFERENCE = REPO_ROOT / 'artifacts' / 'paper_reference' / 'coyote_inference_reference.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect Coyote inference metrics into compact CSV and Markdown tables.')
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--reference-csv', type=Path, default=COYOTE_REFERENCE)
    parser.add_argument('--output-csv', default='coyote-metrics.csv')
    parser.add_argument('--output-md', default='coyote-metrics.md')
    return parser.parse_args()


def read_many(metrics_root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(metrics_root.glob(pattern))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(path) for path in files], how='diagonal')


def warn(message: str) -> None:
    print(f'Warning: {message}', file=sys.stderr)


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
    if value is None:
        return ''
    if isinstance(value, float):
        if math.isnan(value):
            return ''
        return f'{value:.2f}'
    return str(value)


def missing_cell() -> str:
    return '-'


def comparison_table(generated: pl.DataFrame, reference_csv: Path) -> pl.DataFrame:
    columns = ['vertices', 'par', 'batch_size', 'latency_mean_us', 'latency_std_us', 'throughput_mean_samples_per_s']
    generated_table = generated.select([col for col in columns if col in generated.columns])
    if not reference_csv.is_file():
        warn(f'missing Coyote paper reference values: {reference_csv}')
        return generated_table.rename(
            {
                'latency_mean_us': 'latency_mean_us_generated',
                'throughput_mean_samples_per_s': 'throughput_mean_samples_per_s_generated',
            }
        )

    reference = pl.read_csv(reference_csv).rename(
        {
            'latency_mean_us': 'latency_mean_us_paper',
            'throughput_mean_samples_per_s': 'throughput_mean_samples_per_s_paper',
        }
    )
    generated_table = generated_table.rename(
        {
            'latency_mean_us': 'latency_mean_us_generated',
            'throughput_mean_samples_per_s': 'throughput_mean_samples_per_s_generated',
        }
    )
    return reference.join(generated_table, on=['vertices', 'par', 'batch_size'], how='full', coalesce=True)


def write_markdown(table: pl.DataFrame, path: Path) -> None:
    table_rows = [
        '<table>',
        '  <thead>',
        '    <tr><th rowspan="2">Vertices</th><th rowspan="2">PAR</th><th rowspan="2">Batch size</th><th colspan="2">Latency mean (us)</th><th colspan="2">Latency std (us)</th><th colspan="2">Throughput mean (samples/s)</th></tr>',
        '    <tr><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th></tr>',
        '  </thead>',
        '  <tbody>',
    ]
    previous_project: tuple[Any, Any] | None = None
    for row in table.to_dicts():
        project = (row.get('vertices'), row.get('par'))
        if previous_project is not None and project != previous_project:
            table_rows.append('    <tr><td colspan="9"><hr></td></tr>')
        table_rows.append(
            '    <tr>'
            f'<td>{cell(row.get("vertices"))}</td>'
            f'<td>{cell(row.get("par"))}</td>'
            f'<td>{cell(row.get("batch_size"))}</td>'
            f'<td>{cell(row.get("latency_mean_us_paper"))}</td><td>{cell(row.get("latency_mean_us_generated"))}</td>'
            f'<td>{missing_cell()}</td><td>{cell(row.get("latency_std_us"))}</td>'
            f'<td>{cell(row.get("throughput_mean_samples_per_s_paper"))}</td><td>{cell(row.get("throughput_mean_samples_per_s_generated"))}</td>'
            '</tr>'
        )
        previous_project = project
    table_rows.extend(['  </tbody>', '</table>'])
    lines = ['# Coyote Metrics', '', *table_rows]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    args.metrics_root.mkdir(parents=True, exist_ok=True)

    coyote = add_project_columns(read_many(args.metrics_root, 'gravnet_*_coyote_inference.csv'))
    if coyote.is_empty():
        warn(f'no Coyote inference CSV files found in {args.metrics_root} matching gravnet_*_coyote_inference.csv')
        return

    coyote_table = comparison_table(coyote, args.reference_csv).sort(['vertices', 'par', 'batch_size'])
    out_csv = args.metrics_root / args.output_csv
    out_md = args.metrics_root / args.output_md
    coyote_table.write_csv(out_csv)
    write_markdown(coyote_table, out_md)
    print(f'Wrote {out_csv}')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()
