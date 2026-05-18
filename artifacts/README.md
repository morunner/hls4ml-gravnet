# Artifact Evaluation Workflow

This directory contains the scripts and input files used to reproduce the synthesis and end-to-end measurements. The artifact is organized as a small Make-based workflow: first prepare the software environment, then synthesize the designs presented in the paper, and finally measure FPGA inference latency and throughput on an AMD Alveo U55C using the [`CoyoteAccelerator` backend](https://github.com/fpgasystems/Coyote/tree/artefacts/sosp-2025/experiments/07_hls4ml) of [`hls4ml`](https://github.com/fastmachinelearning/hls4ml).

## Step 1: Prepare the Environment

Run:

```bash
make setup-env
```

This installs the Python environment with `uv`, checks that the required tools and Python packages are available, and records the selected Vitis/Vivado paths in `artifacts/.env.mk` for later commands.

**Expected environment:**

- Python 3.11, with dependencies installed through `uv`.
- Xilinx Vitis and Vivado 2025.1.
- Target part `xcu55c-fsvh2892-2L-e`.

The default tool locations for *Vitis HLS* and *Vivado* are:

```text
/tools/Xilinx/2025.1/Vitis
/tools/Xilinx/2025.1/Vivado
```

If your installation uses different paths, pass them when setting up the environment.

```bash
make setup-env \
  VITIS_HLS_PATH=/path/to/Vitis \
  VIVADO_PATH=/path/to/Vivado
```

## Step 2: Run Synthesis

For a quick first pass, synthesize one small configuration:

```bash
make synth VERTICES_PAR=64:1
```

To run the full synthesis sweep of all designs in the paper, run:

```bash
make synth
```

By default, this runs the Coyote backend for:

```text
64:1 64:2 128:1 128:2 256:1 512:1
```

where each entry is `vertices:PAR`. Synthesis products are written under `artifacts/hls4ml_out/`, and the synthesis metrics are collected under `artifacts/metrics/`.

To limit concurrency, set `SYNTH_JOBS`, for example:

```bash
make synth SYNTH_JOBS=1
```

## Step 3: Collect Synthesis Metrics

If synthesis reports already exist, or if you want to regenerate the summary tables without rebuilding the designs, run:

```bash
make synth-metrics
```

This command writes:

```text
artifacts/metrics/synth_metrics.csv
artifacts/metrics/paper_table_synth_comparison.csv
artifacts/metrics/paper_table_synth_comparison.md
```

The comparison table places the paper reference values next to the regenerated values for LUT, DSP, FF, BRAM, cosimulation latency, and cosimulation initiation interval. Missing reports are reported as warnings and leave empty fields in the generated CSV.

## Step 4: Run Coyote Inference

We perform all end-to-end latency and throughput measurements on AMD Alveo U55C cards in the ETH Zurich [Heterogeneous Accelerated Compute Cluster (HACC)](https://github.com/fpgasystems/hacc). To program the FPGA and run the available models:

```bash
make coyote-inference PROGRAM_HACC_FPGA=1
```

The 512-vertex design is intentionally skipped in this step because it does not achieve timing closure, as indicated in the paper. To run a single configuration:

```bash
make coyote-inference VERTICES_PAR=64:1 PROGRAM_HACC_FPGA=1
```

Each run writes one per-design CSV, for example:

```text
artifacts/metrics/gravnet_64vertices_Coyote_1PAR_coyote_inference.csv
```

The default measurement settings are `SAMPLES=4096`, `BATCH_SIZES="1 32 64"`, and `NUM_RUNS=10`. These can be overridden on the command line.

## Step 5: Collect Coyote Inference Metrics

To aggregate the available inference CSV files into a compact table, run:

```bash
make coyote-metrics
```

This writes:

```text
artifacts/metrics/coyote-metrics.csv
artifacts/metrics/coyote-metrics.md
```

The Markdown table groups rows by model configuration for readability.

## Inputs and Outputs

The checked-in model inputs are:

```text
artifacts/models/model_{64,128,256,512}V_cfg.json
artifacts/models/model_{64,128,256,512}V.weights.h5
artifacts/paper_reference/paper_table_reference_values.csv
```

The main generated directories are:

```text
artifacts/hls4ml_out/
artifacts/metrics/
```

To remove all generated outputs and return to a clean artifact workspace, run:

```bash
make clean-artifacts
```
