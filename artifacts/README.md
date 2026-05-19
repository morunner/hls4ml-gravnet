# Artifact Evaluation Workflow

This directory contains the scripts and input files used to reproduce the synthesis and end-to-end measurements. The artifact is organized as a small Make-based workflow: first prepare the software environment, then synthesize the designs presented in the paper, and finally measure FPGA inference latency and throughput on an AMD Alveo U55C using the [`CoyoteAccelerator` backend](https://github.com/fpgasystems/Coyote/tree/artefacts/sosp-2025/experiments/07_hls4ml) of [`hls4ml`](https://github.com/fastmachinelearning/hls4ml).

For a compact overview of the available artifact targets and common options, run:

```bash
make artifact-help
```


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

This artifact uses `uv` to reproduce the Python environment from `uv.lock`. If `uv` is not already available, the recommended installation method is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If your system does not provide `curl`, or if you require alternative installation instructions, consult the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

By default, this runs the syntheses for:

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

### Independent Set-Up

The HACC workflow above is the reference environment used for the measurements in the paper. The same Coyote inference step can also be run on an independent system with an AMD Alveo U55C, provided that Coyote is deployed locally. The general Coyote deployment procedure is described in the [Coyote documentation](https://fpgasystems.github.io/Coyote/intro/quick-start.html#independent-set-up).

The relevant local requirements are:

- AMD Alveo U55C board.
- Linux kernel version 5 or newer.
- CMake 3.5 or newer with C++17 support.
- Vivado Suite, including Vitis HLS, version 2022.1 or newer.
- Huge pages enabled.

To deploy Coyote locally, perform the following steps after synthesis:

1. Program the FPGA with the synthesized bitstream using Vivado Hardware Manager or a custom script.
2. Rescan the PCIe devices and run PCI hot-plug.
3. Insert the driver. The IP and MAC parameters are only needed when using networking on the FPGA:

```bash
sudo insmod Coyote/driver/coyote_driver.ko ip_addr=$qsfp_ip mac_addr=$qsfp_mac
```

A successful completion of the FPGA programming and driver insertion can be checked via a call to:

```bash
dmesg
```

If the driver insertion and bitstream programming completed correctly, the last printed message should be `probe returning 0`. At that point, the system is ready to run the accompanying Coyote software.

If the FPGA is already loaded with a Coyote bitstream and the driver is inserted, make sure to remove the driver with `sudo rmmod coyote_driver` before reprogramming the FPGA with a new bitstream. After programming, the driver can be re-inserted.

Once the FPGA is programmed and the driver is loaded, the artifact command remains the same:

```bash
make coyote-inference PROGRAM_HACC_FPGA=0
```

The `PROGRAM_HACC_FPGA=0` setting avoids the HACC-specific programming helper and assumes that the device has already been prepared locally.

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

## Step 6: Recreate the Paper Plots

To generate the paper plots from newly generated artifact metrics, run:

```bash
make paper-plots
```

This uses `artifacts/metrics/synth_metrics.csv` and `artifacts/metrics/coyote-metrics.csv` where available. GPU benchmark values are taken from the checked-in paper reference data.

To recreate the plots entirely from the checked-in paper reference values, run:

```bash
make paper-plots PLOT_SOURCE=paper
```

Both modes write PNG files under:

```text
artifacts/plots/
```

If PDF output is needed, the plotting script can be run directly with its `--filetype pdf` option, for example:

```bash
uv run --group plotting python artifacts/scripts/plot.py --source paper --filetype pdf
```

## Inputs and Outputs

The checked-in model inputs are:

```text
artifacts/models/model_{64,128,256,512}V_cfg.json
artifacts/models/model_{64,128,256,512}V.weights.h5
artifacts/paper_reference/cosim_reference.csv
artifacts/paper_reference/coyote_inference_reference.csv
artifacts/paper_reference/gpu_benchmarks_reference.csv
artifacts/paper_reference/resource_utilization_reference.csv
```

The main generated directories are:

```text
artifacts/hls4ml_out/
artifacts/metrics/
artifacts/plots/
```

To remove generated metrics and plots, run:

```bash
make artifact-clean
```

The synthesis outputs and saved environment file are kept by default. To remove them as well, run:

```bash
make artifact-clean HARD=1
```
