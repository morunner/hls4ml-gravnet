from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import polars as pl

from common import METRICS_ROOT, REPO_ROOT


REFERENCE = REPO_ROOT / 'artifacts' / 'paper_reference' / 'paper_table_reference_values.csv'
PROJECT_RE = re.compile(r'gravnet_(?P<vertices>\d+)vertices_(?P<backend>[^_]+)_(?P<par>\d+)PAR')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare generated artifact metrics with the paper reference table.')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--output-csv', default='paper_table_comparison.csv')
    parser.add_argument('--output-md', default='paper_table_comparison.md')
    return parser.parse_args()


def parse_project(project: str) -> dict[str, Any]:
    match = PROJECT_RE.search(project)
    if not match:
        return {'vertices': None, 'backend': None, 'par': None}
    return {
        'vertices': int(match.group('vertices')),
        'backend': match.group('backend'),
        'par': int(match.group('par')),
    }


def read_many(metrics_root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(metrics_root.glob(pattern))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(path) for path in files], how='diagonal')


def add_project_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or 'project' not in df.columns:
        return df
    parsed = [parse_project(project) for project in df['project'].to_list()]
    parsed_df = pl.DataFrame(parsed)
    base = df.drop([col for col in ('vertices', 'backend', 'par') if col in df.columns])
    return pl.concat([base, parsed_df], how='horizontal')


def value(row: dict[str, Any], key: str) -> Any:
    val = row.get(key)
    if val == '':
        return None
    return val


def comparison_rows(reference: pl.DataFrame, synth: pl.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    synth_lookup = {}
    for row in synth.to_dicts():
        if row.get('backend') == 'Coyote':
            synth_lookup[(row.get('vertices'), row.get('par'))] = row

    for ref in reference.to_dicts():
        key = (ref['vertices'], ref['par'])
        synth_row = synth_lookup.get(key, {})
        rows.append(
            {
                'model': f'{ref["vertices_label"]} {ref["par"]}PAR',
                'lut_paper': value(ref, 'lut_percent'),
                'lut_generated': value(synth_row, 'lut_percent'),
                'dsp_paper': value(ref, 'dsp_percent'),
                'dsp_generated': value(synth_row, 'dsp_percent'),
                'ff_paper': value(ref, 'ff_percent'),
                'ff_generated': value(synth_row, 'ff_percent'),
                'bram_paper': value(ref, 'bram_percent'),
                'bram_generated': value(synth_row, 'bram_percent'),
                'cosim_latency_paper': value(ref, 'cosim_latency_cycles'),
                'cosim_latency_generated': value(synth_row, 'cosim_latency_min'),
                'cosim_ii_paper': value(ref, 'cosim_ii_cycles'),
                'cosim_ii_generated': value(synth_row, 'cosim_ii_min'),
            }
        )
    return rows


def cell(value_: Any) -> str:
    return '' if value_ is None else str(value_)


def write_markdown(rows: pl.DataFrame, path: Path) -> None:
    table = [
        '<table>',
        '  <thead>',
        '    <tr><th rowspan="2">Model</th><th colspan="2">LUT (%)</th><th colspan="2">DSP (%)</th><th colspan="2">FF (%)</th><th colspan="2">BRAM (%)</th><th colspan="2">Cosim latency (cycles)</th><th colspan="2">Cosim II (cycles)</th></tr>',
        '    <tr><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th><th>Paper</th><th>Generated</th></tr>',
        '  </thead>',
        '  <tbody>',
    ]
    for row in rows.to_dicts():
        table.append(
            '    <tr>'
            f'<td>{cell(row["model"])}</td>'
            f'<td>{cell(row["lut_paper"])}</td><td>{cell(row["lut_generated"])}</td>'
            f'<td>{cell(row["dsp_paper"])}</td><td>{cell(row["dsp_generated"])}</td>'
            f'<td>{cell(row["ff_paper"])}</td><td>{cell(row["ff_generated"])}</td>'
            f'<td>{cell(row["bram_paper"])}</td><td>{cell(row["bram_generated"])}</td>'
            f'<td>{cell(row["cosim_latency_paper"])}</td><td>{cell(row["cosim_latency_generated"])}</td>'
            f'<td>{cell(row["cosim_ii_paper"])}</td><td>{cell(row["cosim_ii_generated"])}</td>'
            '</tr>'
        )
    table.extend(['  </tbody>', '</table>'])
    lines = [
        '# Paper Table Comparison',
        '',
        'This report is informational. It does not enforce exact equality because resource and runtime results can vary by tool installation, license settings, and HACC host state.',
        '',
        *table,
    ]
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    args = parse_args()
    reference = pl.read_csv(args.reference)
    synth = add_project_columns(read_many(args.metrics_root, 'gravnet_*_metrics.csv'))
    if synth.is_empty():
        synth = add_project_columns(read_many(args.metrics_root, 'all_metrics.csv'))
    rows = pl.DataFrame(comparison_rows(reference, synth))

    args.metrics_root.mkdir(parents=True, exist_ok=True)
    out_csv = args.metrics_root / args.output_csv
    out_md = args.metrics_root / args.output_md
    rows.write_csv(out_csv)
    write_markdown(rows, out_md)
    print(f'Wrote {out_csv}')
    print(f'Wrote {out_md}')


if __name__ == '__main__':
    main()
