VERTICES_PAR ?= 64:1 64:2 128:1 128:2 256:1 512:1
BACKENDS ?= CoyoteAccelerator
ARTIFACT_ENV ?= artifacts/.env.mk
PART ?= xcu55c-fsvh2892-2L-e
VITIS_HLS_PATH ?= /tools/Xilinx/2025.1/Vitis
VIVADO_PATH ?= /tools/Xilinx/2025.1/Vivado
DRY_RUN ?= 0
SYNTH_JOBS ?= 0
SAMPLES ?= 4096
BATCH_SIZES ?= 1 32 64
NUM_RUNS ?= 10
PROGRAM_HACC_FPGA ?= 1
PLOT_SOURCE ?= generated
PLOT_NAME_SUFFIX ?= $(PLOT_SOURCE)
PLOT_TITLE ?= $(if $(filter paper,$(PLOT_SOURCE)),Paper reference values,Regenerated artifact results)
HARD ?= 0

.DEFAULT_GOAL := artifact-help

-include $(ARTIFACT_ENV)

DRY_RUN_FLAG = $(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)
PROGRAM_HACC_FPGA_FLAG = $(if $(filter 1 true yes,$(PROGRAM_HACC_FPGA)),--program-hacc-fpga,)
MODEL_ARGS = --models $(VERTICES_PAR)

.PHONY: artifact-help setup-env synth synth-metrics coyote-inference coyote-metrics paper-plots artifact-clean

setup-env:
	@mkdir -p $(dir $(ARTIFACT_ENV))
	@printf 'VITIS_HLS_PATH := %s\nVIVADO_PATH := %s\nPART := %s\n' '$(VITIS_HLS_PATH)' '$(VIVADO_PATH)' '$(PART)' > $(ARTIFACT_ENV)
	uv sync
	uv run python artifacts/scripts/check_environment.py $(MODEL_ARGS) --part $(PART) --vitis-hls-path $(VITIS_HLS_PATH) --vivado-path $(VIVADO_PATH)

synth:
	uv run python artifacts/scripts/synth_all.py $(MODEL_ARGS) --backends $(BACKENDS) --part $(PART) --vitis-hls-path $(VITIS_HLS_PATH) --vivado-path $(VIVADO_PATH) --jobs $(SYNTH_JOBS) $(DRY_RUN_FLAG)
	@if [ "$(DRY_RUN)" != "1" ] && [ "$(DRY_RUN)" != "true" ] && [ "$(DRY_RUN)" != "yes" ]; then \
		uv run python artifacts/scripts/collect_synth_metrics.py $(MODEL_ARGS) --backends $(BACKENDS); \
	fi

synth-metrics:
	uv run python artifacts/scripts/collect_synth_metrics.py $(MODEL_ARGS) --backends $(BACKENDS)
	uv run python artifacts/scripts/compare_reference.py

coyote-inference:
	@for model in $(VERTICES_PAR); do \
		v=$${model%%:*}; par=$${model##*:}; \
		if [ "$$v" = "512" ]; then \
			echo "Skipping $$model for Coyote inference because 512 vertices does not pass timing"; \
			continue; \
		fi; \
		echo "\n\nRunning Coyote inference for $$model..."; \
		uv run python artifacts/scripts/coyote_inference.py --vertices $$v --par $$par --samples $(SAMPLES) --batch-sizes $(BATCH_SIZES) --num-runs $(NUM_RUNS) $(PROGRAM_HACC_FPGA_FLAG); \
	done

coyote-metrics:
	uv run python artifacts/scripts/make_tables.py

paper-plots:
	uv run --group plotting python artifacts/scripts/plot.py --source $(PLOT_SOURCE) --name-suffix $(PLOT_NAME_SUFFIX) --title "$(PLOT_TITLE)"

artifact-clean:
	rm -rf artifacts/metrics artifacts/plots artifacts/scripts/__pycache__
	@if [ "$(HARD)" = "1" ] || [ "$(HARD)" = "true" ] || [ "$(HARD)" = "yes" ]; then \
		rm -rf artifacts/hls4ml_out $(ARTIFACT_ENV); \
	fi

artifact-help:
	@printf 'FPL artifact workflow\n\n'
	@printf 'setup-env\n'
	@printf '  Description: Install/check Python dependencies and record local tool paths.\n'
	@printf '  Usage:       make setup-env [VITIS_HLS_PATH=...] [VIVADO_PATH=...] [PART=...] [VERTICES_PAR="..."]\n'
	@printf '  Options:\n'
	@printf '    VITIS_HLS_PATH=/tools/Xilinx/2025.1/Vitis       Vitis installation path\n'
	@printf '    VIVADO_PATH=/tools/Xilinx/2025.1/Vivado         Vivado installation path\n'
	@printf '    PART=xcu55c-fsvh2892-2L-e                       FPGA part\n'
	@printf '    VERTICES_PAR="64:1 64:2 128:1 128:2 256:1 512:1" Model list as vertices:PAR\n\n'
	@printf 'synth\n'
	@printf '  Description: Build selected HLS/Coyote projects and collect synthesis metrics.\n'
	@printf '  Usage:       make synth [VERTICES_PAR="..."] [BACKENDS=...] [SYNTH_JOBS=...] [DRY_RUN=1]\n'
	@printf '  Options:\n'
	@printf '    VERTICES_PAR="64:1 64:2 128:1 128:2 256:1 512:1" Model list as vertices:PAR\n'
	@printf '    BACKENDS=CoyoteAccelerator                       Backend(s) to synthesize\n'
	@printf '    SYNTH_JOBS=0                                     Concurrent syntheses; 0 means all\n'
	@printf '    DRY_RUN=0                                        Set to 1/true/yes to print commands only\n'
	@printf '    PART=xcu55c-fsvh2892-2L-e                       FPGA part\n'
	@printf '    VITIS_HLS_PATH=/tools/Xilinx/2025.1/Vitis       Vitis installation path\n'
	@printf '    VIVADO_PATH=/tools/Xilinx/2025.1/Vivado         Vivado installation path\n\n'
	@printf 'synth-metrics\n'
	@printf '  Description: Recollect synthesis metrics and compare them with paper reference values.\n'
	@printf '  Usage:       make synth-metrics [VERTICES_PAR="..."] [BACKENDS=...]\n'
	@printf '  Options:\n'
	@printf '    VERTICES_PAR="64:1 64:2 128:1 128:2 256:1 512:1" Model list as vertices:PAR\n'
	@printf '    BACKENDS=CoyoteAccelerator                       Backend(s) to include\n\n'
	@printf 'coyote-inference\n'
	@printf '  Description: Run Coyote FPGA inference for timing-clean models; 512 vertices is skipped.\n'
	@printf '  Usage:       make coyote-inference [VERTICES_PAR="..."] [SAMPLES=...] [BATCH_SIZES="..."] [NUM_RUNS=...] [PROGRAM_HACC_FPGA=1]\n'
	@printf '  Options:\n'
	@printf '    VERTICES_PAR="64:1 64:2 128:1 128:2 256:1 512:1" Model list as vertices:PAR\n'
	@printf '    SAMPLES=4096                                     Synthetic inference samples\n'
	@printf '    BATCH_SIZES="1 32 64"                            Inference batch sizes\n'
	@printf '    NUM_RUNS=10                                      Repetitions per batch size\n'
	@printf '    PROGRAM_HACC_FPGA=1                              Set to 1/true/yes to program HACC FPGA first\n\n'
	@printf 'coyote-metrics\n'
	@printf '  Description: Aggregate available Coyote inference CSVs and compare with paper reference values.\n'
	@printf '  Usage:       make coyote-metrics\n'
	@printf '  Options:     none\n\n'
	@printf 'paper-plots\n'
	@printf '  Description: Generate paper plots from regenerated artifact metrics or checked-in paper references.\n'
	@printf '  Usage:       make paper-plots [PLOT_SOURCE=generated|paper]\n'
	@printf '  Options:\n'
	@printf '    PLOT_SOURCE=generated                            Input data source: generated or paper\n\n'
	@printf 'artifact-clean\n'
	@printf '  Description: Remove generated metrics and plots; optionally remove synthesis outputs and saved env.\n'
	@printf '  Usage:       make artifact-clean [HARD=1]\n'
	@printf '  Options:\n'
	@printf '    HARD=0                                           Set to 1/true/yes to also remove artifacts/hls4ml_out and artifacts/.env.mk\n\n'
