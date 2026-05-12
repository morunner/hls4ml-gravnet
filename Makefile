VERTICES_PAR ?= 64:1 64:2 128:1 128:2 256:1 512:1
BACKENDS ?= CoyoteAccelerator
ARTIFACT_ENV ?= artifacts/.env.mk
PART ?= xcu55c-fsvh2892-2L-e
VITIS_HLS_PATH ?= /tools/Xilinx/2025.1/Vitis
VIVADO_PATH ?= /tools/Xilinx/2025.1/Vivado
DRY_RUN ?= 0
ALLOW_MISSING ?= 0
SYNTH_JOBS ?= 0
SAMPLES ?= 4096
BATCH_SIZES ?= 1 16 32 64
NUM_RUNS ?= 10
PROGRAM_FPGA ?= 0

-include $(ARTIFACT_ENV)

DRY_RUN_FLAG = $(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)
ALLOW_MISSING_FLAG = $(if $(filter 1 true yes,$(ALLOW_MISSING)),--allow-missing,)
PROGRAM_FPGA_FLAG = $(if $(filter 1 true yes,$(PROGRAM_FPGA)),--program-fpga,)
MODEL_ARGS = --models $(VERTICES_PAR)

.PHONY: artifact-setup-env artifact-synth artifact-metrics artifact-coyote artifact-tables artifact-compare artifact-clean

artifact-setup-env:
	@mkdir -p $(dir $(ARTIFACT_ENV))
	@printf 'VITIS_HLS_PATH := %s\nVIVADO_PATH := %s\nPART := %s\n' '$(VITIS_HLS_PATH)' '$(VIVADO_PATH)' '$(PART)' > $(ARTIFACT_ENV)
	uv sync
	uv run python artifacts/scripts/check_environment.py $(MODEL_ARGS) --part $(PART) --vitis-hls-path $(VITIS_HLS_PATH) --vivado-path $(VIVADO_PATH)

artifact-synth:
	uv run python artifacts/scripts/synth_all.py $(MODEL_ARGS) --backends $(BACKENDS) --part $(PART) --vitis-hls-path $(VITIS_HLS_PATH) --vivado-path $(VIVADO_PATH) --jobs $(SYNTH_JOBS) $(DRY_RUN_FLAG)
	@if [ "$(DRY_RUN)" != "1" ] && [ "$(DRY_RUN)" != "true" ] && [ "$(DRY_RUN)" != "yes" ]; then \
		uv run python artifacts/scripts/collect_synth_metrics.py $(MODEL_ARGS) --backends $(BACKENDS) $(ALLOW_MISSING_FLAG); \
	fi

artifact-metrics:
	uv run python artifacts/scripts/collect_synth_metrics.py $(MODEL_ARGS) --backends $(BACKENDS) $(ALLOW_MISSING_FLAG)

artifact-coyote:
	@for model in $(VERTICES_PAR); do \
		v=$${model%%:*}; par=$${model##*:}; \
		uv run python artifacts/scripts/coyote_inference.py --vertices $$v --par $$par --samples $(SAMPLES) --batch-sizes $(BATCH_SIZES) --num-runs $(NUM_RUNS) $(PROGRAM_FPGA_FLAG); \
	done

artifact-tables:
	uv run python artifacts/scripts/make_tables.py

artifact-compare:
	uv run python artifacts/scripts/compare_reference.py

artifact-clean:
	rm -rf artifacts/hls4ml_out artifacts/metrics artifacts/scripts/__pycache__ $(ARTIFACT_ENV)
