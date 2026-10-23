# GPUBackendTools: Backend tools for GPU/CPU agnostic codes.

`gpubackendtools` is the backend-dispatch framework of the LISA Analysis Tools stack
(LISAanalysistools, GBGPU, BBHx, FastEMRIWaveforms and Eryn build on it). It provides:

- one process-wide backend registry, `Globals().backends_manager`, in which every package
  registers its compute backends under `<pkg>_<flavor>` names (`gbt_cpu`,
  `lisatools_cuda12x`, ...), with flavors `cpu` (NumPy), `cuda11x` / `cuda12x` / `cuda13x`
  (CuPy) and `jax`;
- the `Backend` base classes with the availability checks (plugin module installed, CuPy
  usable, CUDA driver version, NVIDIA shared libraries);
- `ParallelModuleBase`, the base of every class that picks its backend once, at
  construction, with `force_backend=`;
- a reference compute object, `gpubackendtools.interpolate.CubicSplineInterpolant` (batched
  not-a-knot cubic splines; LAPACKE fit on CPU, cuSPARSE on CUDA, scipy-fit + JAX
  evaluation on `gbt_jax`), and the C++ / CUDA headers shared by the downstream packages
  (`gbt_global.h`, `cuda_complex.hpp`, `InterpolateDevice.hh`).

## Getting Started

```python
import numpy as np

from gpubackendtools import get_backend, has_backend
from gpubackendtools.interpolate import CubicSplineInterpolant

if has_backend("gbt_cpu"):
    xp = get_backend("gbt_cpu").xp  # numpy

x = np.tile(np.linspace(0.0, 1.0, 100), (2, 1))  # 2 splines of 100 knots
y = np.stack([np.sin(2 * np.pi * x[0]), np.cos(2 * np.pi * x[1])])
spl = CubicSplineInterpolant(x, y, force_backend="cpu")  # or "cuda12x", "gpu", "jax", ...
y_new = spl(np.tile(np.linspace(0.1, 0.9, 7), (2, 1)))  # shape (2, 7)
```

### Prerequisites

- Python >= 3.10.
- To build from source: a C++ compiler and a Fortran compiler (the CMake project enables
  both), and LAPACKE for the CPU backend. LAPACKE is looked up with pkg-config or CMake and
  otherwise downloaded and built (CMake options `GBT_LAPACKE_DETECT_WITH`,
  `GBT_LAPACKE_FETCH`, `GBT_LAPACKE_EXTRA_LIBS`).
- For the CUDA backends: the CUDA toolkit at build time, and at run time a CuPy package for
  the same CUDA major version (`cupy-cuda12x`, ...). CuPy is not a declared dependency.
- Optional: JAX, which enables the pure-JAX backend `gbt_jax`.

### Installing

From a source checkout:

```sh
pip install .                 # release-style install
pip install -e '.[testing]'   # editable development install
```

The CMake option `GBT_WITH_GPU` (`AUTO` default, `ON`, `OFF`, `ONLY`, `BARE`) selects the
compiled backends; with `AUTO` the CUDA backend is built when `nvcc` and the CUDA toolkit are
found. `GBT_CUDA_ARCH` (default `native`) sets the CUDA architectures. Pass options through
pip, e.g. `pip install . --config-settings=cmake.define.GBT_WITH_GPU=OFF`. The compiled
backends are installed as the modules `gbt_backend_cpu` and `gbt_backend_cuda<major>x`.
When a CUDA backend's module is missing, the loader suggests the plugin package
`gpubackendtools-cuda<major>x` (`gpubackendtools-cuda11x`, `-cuda12x`, `-cuda13x`).

## Running the Tests

The tests use `unittest` (the `[testing]` extra installs `matplotlib`):

```sh
python -m unittest discover
python -m unittest tests.test_cubic_spline.CubicSplineTest
```

The GPU code paths are exercised only when CuPy imports.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/lisa-analysis-tools/GPUBackendTools/tags).

## Authors

* Michael Katz
* Maxime Pigou
* Mathieu Dubois

## Contributors


## License

This project is licensed under the Apache License - see the [LICENSE](https://github.com/lisa-analysis-tools/GPUBackendTools/blob/main/LICENSE) file for details.
