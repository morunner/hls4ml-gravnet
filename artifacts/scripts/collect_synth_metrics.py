from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from common import BACKENDS, HLS_ROOT, METRICS_ROOT, VERTICES, parse_model_specs, project_name


class MissingReportError(RuntimeError):
    pass


def parse_coyote_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {}
    build_dir = path / 'build'
    hw_dirs = list(build_dir.glob('*_cyt_hw'))
    if hw_dirs:
        util = hw_dirs[0] / 'reports' / 'shell_utilization.rpt'
        if util.exists():
            report['VivadoSynthReport'] = parse_shell_utilization(util)

    sol = (
        build_dir
        / f'{path.name}_cyt_hw'
        / f'{path.name}_config_0'
        / 'user_c0_0/hdl/ext/model_wrapper_hls/model_wrapper_c0_0/solution1'
    )
    report['CosimReport'] = parse_cosim_transaction(sol) or parse_cosim_report(sol)
    return report


def parse_shell_utilization(path: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    with path.open() as f:
        for line in f:
            if not line.strip().startswith('|'):
                continue
            parts = [x.strip() for x in line.split('|') if x.strip()]
            if len(parts) < 3:
                continue
            name, used = parts[0], parts[1]
            available = parts[-2] if len(parts) > 3 and number(parts[-2]) is not None else None
            if 'CLB LUTs' in name or 'Slice LUTs' in name:
                rows.setdefault('LUT', used)
                rows.setdefault('AvailableLUT', available)
            elif 'CLB Registers' in name or 'Slice Registers' in name:
                rows.setdefault('FF', used)
                rows.setdefault('AvailableFF', available)
            elif 'DSPs' in name:
                rows.setdefault('DSP48E', used)
                rows.setdefault('AvailableDSP', available)
            elif 'RAMB18' in name and 'RAMB18E2' not in name:
                rows.setdefault('BRAM_18K', used)
                rows.setdefault('AvailableBRAM_18K', available)
            elif 'RAMB36' in name and 'RAMB36E2' not in name:
                rows.setdefault('BRAM_36K', used)
    return rows


def parse_cosim_transaction(solution_dir: Path) -> dict[str, Any]:
    files = list(solution_dir.glob('sim/*/*.performance.result.transaction.xml'))
    if not files:
        return {}
    latency: list[int] = []
    interval: list[int] = []
    with files[0].open() as f:
        for line in f:
            if 'transaction' not in line or ':' not in line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                latency.append(int(parts[2]))
            if len(parts) >= 4 and parts[3] != 'x':
                interval.append(int(parts[3]))
    if not latency:
        return {}
    return {
        'LatencyMin': min(latency),
        'LatencyMax': max(latency),
        'LatencyAvg': sum(latency) / len(latency),
        'IntervalMin': min(interval) if interval else None,
        'IntervalMax': max(interval) if interval else None,
        'IntervalAvg': sum(interval) / len(interval) if interval else None,
    }


def parse_cosim_report(solution_dir: Path) -> dict[str, Any]:
    files = list((solution_dir / 'sim/report').glob('*_cosim.rpt')) if (solution_dir / 'sim/report').exists() else []
    if not files:
        return {}
    with files[0].open() as f:
        for line in f:
            if '|' not in line or ('Verilog' not in line and 'VHDL' not in line):
                continue
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 8 and parts[1] != 'NA':
                return {
                    'Status': parts[1],
                    'LatencyMin': parts[2],
                    'LatencyMax': parts[4],
                    'IntervalMin': parts[5],
                    'IntervalMax': parts[7],
                }
    return {}


def parse_report(path: Path, backend: str) -> dict[str, Any]:
    if not path.exists():
        raise MissingReportError(f'Missing HLS project directory: {path}')
    if backend == 'CoyoteAccelerator':
        return parse_coyote_report(path)
    from hls4ml.report import parse_vivado_report

    return parse_vivado_report(str(path)) or {}


def number(value: Any) -> float | None:
    try:
        if isinstance(value, str):
            value = value.replace(',', '').replace('%', '').strip()
        return float(value)
    except (TypeError, ValueError):
        return None


def percent(used: float | None, available: float | None) -> int | None:
    if used is None or available in (None, 0):
        return None
    return int(round(used / available * 100))


def validate_report(path: Path, backend: str, report: dict[str, Any]) -> None:
    required_sections = ('VivadoSynthReport', 'CosimReport')
    missing = [section for section in required_sections if not report.get(section)]
    if missing:
        raise MissingReportError(f'Missing required report section(s) for {path}: {", ".join(missing)}')
    vsynth = report['VivadoSynthReport']
    cosim = report['CosimReport']
    required_values = {
        'VivadoSynthReport.LUT': vsynth.get('LUT'),
        'VivadoSynthReport.FF': vsynth.get('FF'),
        'VivadoSynthReport.DSP48E/DSP': vsynth.get('DSP48E', vsynth.get('DSP')),
        'VivadoSynthReport.BRAM_18K': vsynth.get('BRAM_18K'),
        'VivadoSynthReport.AvailableLUT': vsynth.get('AvailableLUT'),
        'VivadoSynthReport.AvailableFF': vsynth.get('AvailableFF'),
        'VivadoSynthReport.AvailableDSP': vsynth.get('AvailableDSP'),
        'VivadoSynthReport.AvailableBRAM_18K': vsynth.get('AvailableBRAM_18K'),
        'CosimReport.LatencyMin': cosim.get('LatencyMin'),
        'CosimReport.IntervalMin': cosim.get('IntervalMin'),
    }
    missing_values = [name for name, value in required_values.items() if value in (None, '')]
    if missing_values:
        raise MissingReportError(f'Missing required report value(s) for {path}: {", ".join(missing_values)}')


def collect_one(path: Path, backend: str, allow_missing: bool = False) -> dict[str, Any]:
    try:
        report = parse_report(path, backend)
    except MissingReportError:
        if not allow_missing:
            raise
        report = {}
    if not allow_missing:
        validate_report(path, backend, report)
    vsynth = report.get('VivadoSynthReport', {})
    cosim = report.get('CosimReport', {})
    lut = number(vsynth.get('LUT'))
    ff = number(vsynth.get('FF'))
    dsp = number(vsynth.get('DSP48E', vsynth.get('DSP')))
    bram_18k = number(vsynth.get('BRAM_18K'))
    bram_36k = number(vsynth.get('BRAM_36K'))
    bram = None if bram_18k is None and bram_36k is None else (bram_18k or 0) + 2 * (bram_36k or 0)
    return {
        'backend': backend,
        'project_dir': str(path),
        'lut': lut,
        'ff': ff,
        'dsp': dsp,
        'bram_18k_equiv': bram,
        'lut_percent': percent(lut, number(vsynth.get('AvailableLUT'))),
        'ff_percent': percent(ff, number(vsynth.get('AvailableFF'))),
        'dsp_percent': percent(dsp, number(vsynth.get('AvailableDSP'))),
        'bram_percent': percent(bram, number(vsynth.get('AvailableBRAM_18K'))),
        'cosim_latency_min': cosim.get('LatencyMin'),
        'cosim_ii_min': cosim.get('IntervalMin'),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Collect resource and cosim metrics from hls4ml reports.')
    parser.add_argument('--vertices', type=int, nargs='+', default=list(VERTICES), choices=VERTICES)
    parser.add_argument('--backends', nargs='+', default=list(BACKENDS), choices=BACKENDS)
    parser.add_argument('--par', type=int, default=1)
    parser.add_argument(
        '--models',
        nargs='+',
        help='Explicit model list as VERTICES:PAR entries, e.g. 64:1 64:2 128:1.',
    )
    parser.add_argument('--tag')
    parser.add_argument('--hls-root', type=Path, default=HLS_ROOT)
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--output', default='all_metrics.csv')
    parser.add_argument('--allow-missing', action='store_true', help='Write partial rows instead of failing on missing reports.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    models = parse_model_specs(args.models) if args.models else [(vertices, args.par) for vertices in args.vertices]
    for vertices, par in models:
        for backend in args.backends:
            stem = project_name(vertices, backend, par, args.tag)
            path = args.hls_root / stem
            rows.append({'vertices': vertices, 'project': stem, **collect_one(path, backend, args.allow_missing)})

    args.metrics_root.mkdir(parents=True, exist_ok=True)
    out_file = args.metrics_root / args.output
    pl.DataFrame(rows).write_csv(out_file)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
