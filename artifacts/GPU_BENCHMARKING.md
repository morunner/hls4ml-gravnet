# GPU Benchmarking Workflow

This note documents the GPU benchmarking workflow used for the comparison baseline in the paper. The GPU results are provided only as a comparison point for the FPGA measurements. They are therefore documented here rather than packaged as a fully reproducible artifact with a custom Docker environment or checked-in model weights.

The benchmark was run for three non-quantized floating-point GravNet models with 64, 128, and 256 vertices that matched the architecture of the quantized models deployed on FPGA. The reported GPU baseline uses FP16 TensorRT engines on an NVIDIA L40S GPU. End-to-end latency includes host-to-device transfer, GPU execution, and device-to-host transfer, matching the FPGA end-to-end measurement convention used in the paper.

## 1. Hardware and software environment

The GPU measurements were run on a host with:

```text
CPU: AMD EPYC 9534 64-Core Processor
GPU: NVIDIA L40S
```

The software environment was based on the official NVIDIA TensorRT container:

```dockerfile
nvcr.io/nvidia/tensorrt:24.01-py3
```

The main tools and libraries used in the workflow were:

```text
TensorRT 8.6.1
trtexec from TensorRT 8.6.1
TensorFlow / Keras 2.14.0
QKeras 0.9.0
tf2onnx 1.17.0
quantized-gravnet
```

TensorRT 10 was also tested for this workload and did not improve performance over TensorRT 8.6.1. The paper therefore reports the best-performing TensorRT 8.6.1 results.

## 2. Train full-precision Keras models and export to ONNX 

The GPU workflow was applied to three full-precision GravNet models with 64, 128 and 256 vertices respectively. The models were instantiated via `GravNetFactory` instances with the same configuration, minus quantization configuration, as the respective `QGravNetFactory` used for the quantized models.  

The full-precision models were converted to ONNX format via `tf2onnx`.

```bash
python -m tf2onnx.convert --saved-model MODEL_PATH --output ONNX_MODEL_PATH --opset 17
```

## 3. Build TensorRT engines and run `trtexec`

For each model size and reported batch size (1, 32, 64), we built a static TensorRT engine:

```bash
trtexec \
  --onnx="<model.onnx>" \
  --shapes="gravnet_input:<BATCH>x<VERTICES>x<FEATURES>" \
  --fp16 \
  --dumpLayerInfo \
  --builderOptimizationLevel=5 \
  --avgTiming=64 \
  --profilingVerbosity=detailed \
  --saveEngine="<output_dir>/fp16_bs<BATCH>_v<VERTICES>.engine" \
  > "<output_dir>/fp16_bs<BATCH>_v<VERTICES>_build.log"
```

After building the engine, we ran `trtexec` benchmarking using synthetic inputs.

```bash
trtexec \
  --loadEngine="<engine>" \
  --warmUp=200 \
  --iterations=2000 \
  --useCudaGraph \
  --useSpinWait \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportTimes="<output_dir>/fp16_bs<BATCH>_v<VERTICES>_e2e.json" \
  > "<output_dir>/fp16_bs<BATCH>_v<VERTICES>_e2e.log"
```

We report mean end-to-end latency from the `trtexec` timing reports. Throughput is computed from the reported QPS multiplied by the batch size.

## 4. Validate TensorRT engine outputs

Each TensorRT engine was validated against Keras reference outputs for functional correctness. The validation was run for every generated engine for the three model sizes and three reported batch sizes.

The validation procedure was:

1. Load the serialized TensorRT engine using the TensorRT Python bindings.
2. Load a fixed test input array and the corresponding Keras reference outputs.
3. Run TensorRT inference batch-by-batch using the static engine batch size.
4. Compare the TensorRT output with the Keras output.

The validation checks completed successfully for all reported TensorRT engines.
