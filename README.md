# hls4ml-gravnet: FPL 2026 Artifacts

This branch contains the artifacts of the paper *Ultra-Low Latency and Scalable GravNet Clustering on FPGAs for Highly Granular Detectors*, submitted to FPL 2026. The branch `artifacts/fpl-2026` adds the source code, model inputs, scripts, and reference tables needed to reproduce the paper results.

Reviewers should start from the artifact workflow in [`artifacts/`](artifacts/README.md). It is organized as six steps: [(1)](artifacts/README.md#step-1-prepare-the-environment) prepare the environment, [(2)](artifacts/README.md#step-2-run-synthesis) run synthesis, [(3)](artifacts/README.md#step-3-collect-synthesis-metrics) collect synthesis metrics, [(4)](artifacts/README.md#step-4-run-coyote-inference) run Coyote inference, [(5)](artifacts/README.md#step-5-collect-coyote-inference-metrics) collect Coyote inference metrics, and [(6)](artifacts/README.md#step-6-recreate-the-paper-plots) recreate the paper plots. Steps 1 to 3 and 6 can be run on any machine with Python, Vivado, and Vitis installed. Step 4 uses AMD Alveo U55C cards on the [ETHZ HACC cluster](https://github.com/fpgasystems/hacc), which is publicly available but requires an account, or an equivalent local U55C setup; see the [independent setup notes](artifacts/README.md#independent-set-up) for the latter.

This branch will remain frozen as the reference point for comparisons with the paper. Future development will continue on the upstream `main` branch and will not affect this artifact branch. Additionally, a copy of the approved artifact is uploaded to Zenodo.

## hls4ml-gravnet

This repository contains the High-Level Synthesis (HLS) headers and Python auxiliaries required to synthesize the quantized GravNet model from [quantized-gravnet](https://github.com/lorenzo-as/quantized-gravnet) using [hls4ml](https://github.com/fastmachinelearning/hls4ml) for real-time inference on FPGAs.

### Key Features & Requirements
* Utilizes the hls4ml [Extension API](https://fastmachinelearning.org/hls4ml/advanced/extension.html) to convert models from Keras to HLS.
* **Experimental Fork:** This project was initially tested with a specific experimental fork of hls4ml located [here](https://github.com/morunner/hls4ml/tree/this-fork-int). This fork includes necessary changes to support:
  * Synthesizing hls4ml with the Coyote accelerator backend.
  * Compatibility with Vitis/Vivado 2025.1, with Coyote.
  * Processing two vertices simultaneously per clock cycle.
  * Supporting multiple outputs with Coyote.

Any feedback for running synthesis with the hls4ml [upstream](https://github.com/fastmachinelearning/hls4ml) is highly welcome.

## Getting Started

### Prerequisites

[uv](https://docs.astral.sh/uv/) is used as the package manager. Make sure it is available on your system before proceeding.

### Installation

To create a virtual environment and install the `hls4ml_gravnet` package, run:

```bash
uv sync
```

If you want to install it as an editable package for active development:

```bash
uv pip install -e .
```

If you already have a project and want to add this package to your existing virtual environment:

```bash
uv add git+https://github.com/fpgasystems/hls4ml-gravnet.git
```

## Usage

Here is an example of converting the `QGravNetFactory` model to HLS using the `Vitis` (or `CoyoteAccelerator`) backend.

```python
import hls4ml
from hls4ml_gravnet.hls4ml_extension.register_extensions import register_extensions

# Register the custom GravNet layers
register_extensions(backend='Vitis') # Or 'CoyoteAccelerator'

# Configure HLS from the Keras model
hls_config = hls4ml.utils.config_from_keras_model(model)

# Note: Tune precisions and reuse factors in the hls_config here as needed.

# Convert the model to an HLS project
hls_model = hls4ml.converters.convert_from_keras_model(
    model=model,
    hls_config=hls_config,
    backend='Vitis',
    output_dir='my_hls_project'
)

# Compile the HLS model for C-simulation
hls_model.compile()

# Synthesize the RTL hardware
hls_model.build()
```

## References

This project implements layers from the original GravNet architecture described in:

> **Learning representations of irregular particle-detector geometry with distance-weighted graph networks**
> Shah Rukh Qasim, Jan Kieseler, Yutaro Iiyama, Maurizio Pierini
> *Eur. Phys. J. C79 (2019) no.7, 608.*
> [[DOI]](https://doi.org/10.1140/epjc/s10052-019-7113-9)

```bibtex
@article{Qasim:2019otl,
      author         = "Qasim, Shah Rukh and Kieseler, Jan and Iiyama, Yutaro and
                        Pierini, Maurizio",
      title          = "{Learning representations of irregular particle-detector
                        geometry with distance-weighted graph networks}",
      journal        = "Eur. Phys. J.",
      volume         = "C79",
      year           = "2019",
      number         = "7",
      pages          = "608",
      doi            = "10.1140/epjc/s10052-019-7113-9",
      eprint         = "1902.07987",
      archivePrefix  = "arXiv",
      primaryClass   = "physics.data-an",
      SLACcitation   = "%%CITATION = ARXIV:1902.07987;%%"
}
```

The hardware implementations in this repository succeeded thanks to the foundational work and architectural designs from:

> **Real-Time Graph-based Point Cloud Networks on FPGAs via Stall-Free Deep Pipelining**
> Marc Neu, Isabel Haide, Timo Justinger, Till Rädler, Valdrin Dajaku, Torben Ferber, Jürgen Becker
> *arXiv preprint (2025).*
> [[arXiv:2507.05099]](https://arxiv.org/abs/2507.05099)

```bibtex
@misc{neu2025,
      title={Real-Time Graph-based Point Cloud Networks on FPGAs via Stall-Free Deep Pipelining},
      author={Marc Neu and Isabel Haide and Timo Justinger and Till Rädler and Valdrin Dajaku and Torben Ferber and Jürgen Becker},
      year={2025},
      eprint={2507.05099},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2507.05099},
}
```
