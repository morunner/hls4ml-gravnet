from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mplhep
import numpy as np
import polars as pl

from common import METRICS_ROOT, REPO_ROOT


REFERENCE_ROOT = REPO_ROOT / 'artifacts' / 'paper_reference'
PLOTS_ROOT = REPO_ROOT / 'artifacts' / 'plots'
PROJECT_RE = re.compile(r'gravnet_(?P<vertices>\d+)vertices_(?P<backend>[^_]+)_(?P<par>\d+)PAR')

COLORS = {'fpga_par1': '#4477AA', 'fpga_par2': '#EE6677', 'gpu': '#228833'}
FPGA_HANDLES = [
    patches.Patch(facecolor=COLORS['fpga_par1'], edgecolor='black', label='PAR1'),
    patches.Patch(facecolor=COLORS['fpga_par2'], edgecolor='black', label='PAR2'),
]
ALL_HANDLES = FPGA_HANDLES + [patches.Patch(facecolor=COLORS['gpu'], alpha=0.8, edgecolor='black', label='GPU')]

IMG_FILETYPES = Literal['png', 'pdf']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate paper plots from generated artifact metrics or paper reference data.')
    parser.add_argument('--source', choices=('generated', 'paper'), default='generated')
    parser.add_argument('--metrics-root', type=Path, default=METRICS_ROOT)
    parser.add_argument('--reference-root', type=Path, default=REFERENCE_ROOT)
    parser.add_argument('--output-root', type=Path, default=PLOTS_ROOT)
    parser.add_argument('--filetype', choices=IMG_FILETYPES, default='png')
    parser.add_argument('--name-suffix', default=None, help='Suffix appended to each plot filename stem.')
    parser.add_argument('--title', default=None, help='Optional figure suptitle shown above each plot.')
    return parser.parse_args()


def configure_style() -> None:
    mplhep.style.use('ATLAS')
    plt.rcParams['figure.figsize'] = (8, 5)
    mpl.rcParams['xtick.minor.visible'] = False


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
    parsed = pl.DataFrame([parse_project(project) for project in df['project'].to_list()])
    base = df.drop([col for col in ('vertices', 'backend', 'par') if col in df.columns])
    return pl.concat([base, parsed], how='horizontal')


def read_required(path: Path) -> pl.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Missing plot input {path}')
    return pl.read_csv(path)


def read_many(metrics_root: Path, pattern: str) -> pl.DataFrame:
    files = sorted(metrics_root.glob(pattern))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_csv(path) for path in files], how='diagonal')


def normalize_backend(df: pl.DataFrame) -> pl.DataFrame:
    if 'backend' not in df.columns:
        return df
    return df.with_columns(pl.col('backend').str.replace('CoyoteAccelerator', 'Coyote'))


def use_generated_coyote_columns(df: pl.DataFrame) -> pl.DataFrame:
    renames = {
        'latency_mean_us_generated': 'latency_mean_us',
        'throughput_mean_samples_per_s_generated': 'throughput_mean_samples_per_s',
    }
    return df.rename({src: dst for src, dst in renames.items() if src in df.columns and dst not in df.columns})


def load_synth_generated(metrics_root: Path) -> pl.DataFrame:
    synth_path = metrics_root / 'synth_metrics.csv'
    synth = pl.read_csv(synth_path) if synth_path.is_file() else read_many(metrics_root, 'gravnet_*_metrics.csv')
    if synth.is_empty():
        return synth
    return normalize_backend(add_project_columns(synth))


def load_inputs(source: str, metrics_root: Path, reference_root: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    gpu = read_required(reference_root / 'gpu_benchmarks_reference.csv')
    if source == 'paper':
        coyote = read_required(reference_root / 'coyote_inference_reference.csv')
        resources = read_required(reference_root / 'resource_utilization_reference.csv')
        cosim = read_required(reference_root / 'cosim_reference.csv')
    else:
        coyote = use_generated_coyote_columns(read_required(metrics_root / 'coyote-metrics.csv'))
        synth = load_synth_generated(metrics_root)
        if synth.is_empty():
            raise FileNotFoundError(f'Missing generated synthesis metrics in {metrics_root}')
        resources = synth.select(
            ['backend', 'vertices', 'par', 'lut_percent', 'ff_percent', 'dsp_percent', 'bram_percent']
        )
        cosim = synth.select(['backend', 'vertices', 'par', 'cosim_latency_min', 'cosim_ii_min'])
    return coyote, gpu, resources, cosim


def scalar(df: pl.DataFrame, filters: dict[str, Any], col: str) -> Any:
    filtered = df
    for name, value in filters.items():
        filtered = filtered.filter(pl.col(name) == value)
    if filtered.is_empty() or col not in filtered.columns:
        return None
    return filtered[0, col]


def draw_bar(ax: Any, x: float, height: Any, width: float, color: str, zorder: int = 3) -> None:
    if height is None or (isinstance(height, float) and np.isnan(height)):
        return
    ax.bar(x, height, width=width, color=color, edgecolor='black', linewidth=0.8, zorder=zorder)


def draw_cluster(ax: Any, center: float, entries: list[dict[str, Any]], width: float) -> None:
    offsets = (np.arange(len(entries)) - (len(entries) - 1) / 2.0) * width
    for offset, entry in zip(offsets, entries):
        draw_bar(ax, center + offset, entry['height'], width, entry['color'], entry.get('zorder', 3))


def apply_title(fig: Any, title: str | None) -> None:
    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    else:
        fig.tight_layout()


def plot_stem(stem: str, suffix: str | None) -> str:
    return f'{stem}_{suffix}' if suffix else stem


def save(fig: Any, output_root: Path, stem: str, filetype: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_root / f'{stem}.{filetype}', bbox_inches='tight')
    print(f'Wrote {output_root / f"{stem}.{filetype}"}')
    plt.close(fig)


def plot_e2e(
    coyote: pl.DataFrame,
    gpu: pl.DataFrame,
    output_root: Path,
    batch_sizes: tuple[int, int] = (32, 64),
    filetype: IMG_FILETYPES = 'png',
    name_suffix: str | None = None,
    title: str | None = None,
) -> None:
    vertices = gpu['vertices'].drop_nulls().unique().sort().to_list()
    x = np.arange(len(vertices))
    xlim_margin = 0.5

    fig, ax_lat = plt.subplots()
    for i, vertices_count in enumerate(vertices):
        entries = [
            {
                'height': scalar(coyote, {'vertices': vertices_count, 'par': 1, 'batch_size': 1}, 'latency_mean_us'),
                'color': COLORS['fpga_par1'],
            }
        ]
        if vertices_count in (64, 128):
            entries.append(
                {
                    'height': scalar(coyote, {'vertices': vertices_count, 'par': 2, 'batch_size': 1}, 'latency_mean_us'),
                    'color': COLORS['fpga_par2'],
                }
            )
        entries.append(
            {
                'height': scalar(gpu, {'vertices': vertices_count, 'batch_size': 1}, 'latency_mean_us'),
                'color': COLORS['gpu'],
                'zorder': 1,
            }
        )
        draw_cluster(ax_lat, x[i], entries, 0.15)

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(vertices)
    ax_lat.set_xlabel('Number of Vertices')
    ax_lat.set_ylabel(r'Latency ($\mathrm{\mu s}$)')
    ax_lat.set_yscale('log')
    ax_lat.set_ylim(1, 1e3)
    ax_lat.set_xlim(x[0] - xlim_margin, x[-1] + xlim_margin)
    for i in range(len(x) - 1):
        ax_lat.axvline((x[i] + x[i + 1]) / 2, color='grey', linestyle=':', alpha=0.5, zorder=1)
    ax_lat.legend(handles=ALL_HANDLES, loc='upper right', frameon=False)
    apply_title(fig, title)
    save(fig, output_root, plot_stem('latency_over_vertices_BS1', name_suffix), filetype)

    fig, axs_thr = plt.subplots(nrows=1, ncols=2, sharey=True)
    for ax_thr, batch_size in zip(axs_thr, batch_sizes):
        for i, vertices_count in enumerate(vertices):
            entries = [
                {
                    'height': scalar(
                        coyote, {'vertices': vertices_count, 'par': 1, 'batch_size': batch_size}, 'throughput_mean_samples_per_s'
                    ),
                    'color': COLORS['fpga_par1'],
                }
            ]
            if vertices_count in (64, 128):
                entries.append(
                    {
                        'height': scalar(
                            coyote,
                            {'vertices': vertices_count, 'par': 2, 'batch_size': batch_size},
                            'throughput_mean_samples_per_s',
                        ),
                        'color': COLORS['fpga_par2'],
                    }
                )
            entries.append(
                {
                    'height': scalar(gpu, {'vertices': vertices_count, 'batch_size': batch_size}, 'throughput_mean_samples_per_s'),
                    'color': COLORS['gpu'],
                    'zorder': 1,
                }
            )
            draw_cluster(ax_thr, x[i], entries, 0.2)

        ax_thr.set_xticks(x)
        ax_thr.set_xticklabels(vertices)
        ax_thr.set_xlim(x[0] - xlim_margin * 1.05, x[-1] + xlim_margin)
        ax_thr.set_ylim(0, 550e3)
        if ax_thr == axs_thr[0]:
            ax_thr.set_ylabel('Throughput (Samples/s)')
        if ax_thr == axs_thr[-1]:
            ax_thr.set_xlabel('Number of Vertices')
        ax_thr.ticklabel_format(axis='y', style='sci', scilimits=(3, 3))
        for i in range(len(x) - 1):
            ax_thr.axvline((x[i] + x[i + 1]) / 2, color='grey', linestyle=':', alpha=0.5, zorder=1)
        ax_thr.legend(handles=ALL_HANDLES, loc='upper right', bbox_to_anchor=(0.98, 0.97), frameon=False)
        mplhep.add_text(f'Batch Size {batch_size}', loc='upper right', ax=ax_thr, fontweight='bold')
    apply_title(fig, title)
    save(fig, output_root, plot_stem(f'throughput_over_vertices_BS{"_".join(str(batch_size) for batch_size in batch_sizes)}', name_suffix), filetype)


def plot_resources(
    resources: pl.DataFrame,
    output_root: Path,
    backend: str = 'Coyote',
    filetype: Literal['png', 'pdf'] = 'png',
    name_suffix: str | None = None,
    title: str | None = None,
) -> None:
    df = resources.filter(pl.col('backend') == backend)
    vertices = df['vertices'].drop_nulls().unique().sort().to_list()
    x = np.arange(len(vertices))
    resource_cols = [
        ('lut_percent', 'LUT'),
        ('ff_percent', 'FF'),
        ('dsp_percent', 'DSP'),
        ('bram_percent', 'BRAM'),
    ]
    fig, axs = plt.subplots(nrows=2, ncols=2, sharey=False, gridspec_kw={'height_ratios': [1, 1]})
    for ax, (column, label) in zip(axs.flatten(), resource_cols):
        for i, vertices_count in enumerate(vertices):
            entries = [
                {
                    'height': scalar(df, {'vertices': vertices_count, 'par': 1}, column),
                    'color': COLORS['fpga_par1'],
                }
            ]
            par2_value = scalar(df, {'vertices': vertices_count, 'par': 2}, column)
            if par2_value is not None:
                entries.append({'height': par2_value, 'color': COLORS['fpga_par2']})
            draw_cluster(ax, x[i], entries, 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(vertices)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(0, np.ceil(df[column].drop_nulls().max() / 10 + 1) * 10)
        ax.set_ylabel(f'{label} (%)')
        if ax in axs.flatten()[-1:]:
            ax.set_xlabel('Number of Vertices')
        for i in range(len(x) - 1):
            ax.axvline((x[i] + x[i + 1]) / 2, color='grey', linestyle=':', alpha=0.5, zorder=1)
    fig.legend(handles=FPGA_HANDLES, loc='upper center', bbox_to_anchor=(0.5, 0.95), frameon=False, ncol=len(FPGA_HANDLES))
    apply_title(fig, title)
    save(fig, output_root, plot_stem(f'resource_utilization_{backend.lower()}', name_suffix), filetype)


def plot_cosim(
    cosim: pl.DataFrame,
    output_root: Path,
    backend: str = 'Coyote',
    filetype: IMG_FILETYPES = 'png',
    name_suffix: str | None = None,
    title: str | None = None,
) -> None:
    df = cosim.filter(pl.col('backend') == backend)
    vertices = df['vertices'].drop_nulls().unique().sort().to_list()
    x = np.arange(len(vertices))
    metrics = [('cosim_latency_min', 'Latency (clock cycles)'), ('cosim_ii_min', 'II (clock cycles)')]
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=False)
    for ax, (column, ylabel) in zip(axs, metrics):
        for i, vertices_count in enumerate(vertices):
            entries = [
                {
                    'height': scalar(df, {'vertices': vertices_count, 'par': 1}, column),
                    'color': COLORS['fpga_par1'],
                }
            ]
            par2_value = scalar(df, {'vertices': vertices_count, 'par': 2}, column)
            if par2_value is not None:
                entries.append({'height': par2_value, 'color': COLORS['fpga_par2']})
            draw_cluster(ax, x[i], entries, 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels(vertices)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(0, None)
        ax.set_ylabel(ylabel)
        if ax == axs[-1]:
            ax.set_xlabel('Number of Vertices')
        for i in range(len(x) - 1):
            ax.axvline((x[i] + x[i + 1]) / 2, color='grey', linestyle=':', alpha=0.5, zorder=1)
        ax.legend(handles=FPGA_HANDLES, loc='upper left', bbox_to_anchor=(0.0, 1.04), frameon=False)
    apply_title(fig, title)
    save(fig, output_root, plot_stem(f'cosim_results_{backend.lower()}', name_suffix), filetype)


def main() -> None:
    args = parse_args()
    configure_style()
    coyote, gpu, resources, cosim = load_inputs(args.source, args.metrics_root, args.reference_root)
    name_suffix = args.name_suffix if args.name_suffix is not None else args.source
    plot_e2e(coyote, gpu, args.output_root, filetype=args.filetype, name_suffix=name_suffix, title=args.title)
    plot_resources(resources, args.output_root, backend='Coyote', filetype=args.filetype, name_suffix=name_suffix, title=args.title)
    plot_cosim(cosim, args.output_root, backend='Coyote', filetype=args.filetype, name_suffix=name_suffix, title=args.title)


if __name__ == '__main__':
    main()
