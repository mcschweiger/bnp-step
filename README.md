# bnp-step

:construction: This page is actively under construction! Check back often for updates. :construction:

This repository contains the Julia package and helper functions for BNP-Step, a computational method described in [An accurate probabilistic step finder for time-series analysis, bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.09.19.558535v1).

## Installation

BNP-Step is now a Julia package. To install and set up the package, run the following commands in the Julia REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Alternatively, you can clone the repository and add it as a package:

```julia
git clone https://github.com/mcschweiger/bnp-step.git
cd bnp-step
using Pkg
Pkg.develop(path=".")
Pkg.instantiate()
```

BNP-Step includes GPU support via `CUDA.jl`. Ensure that your system has a compatible NVIDIA GPU and the necessary CUDA drivers installed.

## Usage

Once installed, you can use BNP-Step by importing the package:

```julia
using BNPStep

# Example usage
BNPStep.runBNPStep(data)
BNPStep.loadTutorial()
```

In the near future, we plan to add an option for running BNP-Step using a simple GUI.

## GPU Support

BNP-Step supports GPU acceleration for computationally intensive tasks. If a compatible GPU is detected, computations will automatically leverage GPU resources. For users without a GPU, the code will gracefully fall back to CPU execution.

## Questions? Contact us!

BNP-Step is a work in progress. Further documentation will be provided as it is created. If you require assistance or would like more details, please do not hesitate to contact us at mcschwei@asu.edu or spresse@asu.edu.

## Acknowledgments

This updated version was developed by Max Schweiger, based on his work with A. Rojewski and S. Presse, with significant contributions from members of Garcia Lab at Berkley as well as GitHub Copilot. We thank all contributors for their efforts in improving BNP-Step.
