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
PROGRAM_HACC_FPGA ?= 0

-include $(ARTIFACT_ENV)

DRY_RUN_FLAG = $(if $(filter 1 true yes,$(DRY_RUN)),--dry-run,)
PROGRAM_HACC_FPGA_FLAG = $(if $(filter 1 true yes,$(PROGRAM_HACC_FPGA)),--program-hacc-fpga,)
MODEL_ARGS = --models $(VERTICES_PAR)

.PHONY: setup-env synth synth-metrics coyote-inference coyote-metrics clean-artifacts clean

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

clean-artifacts:
	rm -rf artifacts/hls4ml_out artifacts/metrics artifacts/scripts/__pycache__ $(ARTIFACT_ENV)

clean: clean-artifacts
