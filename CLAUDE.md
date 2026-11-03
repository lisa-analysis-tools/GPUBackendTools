# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**LISA Analysis Tools–wide conventions:** [`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md) (canonical). **This repo's map:** [`docs/codebase-map.md`](docs/codebase-map.md).

## Project Overview

`gpubackendtools` (imported as `gpubackendtools`) is the **backend dispatch framework** that downstream LISA/GW packages (e.g. `lisaanalysistools`, `fastemriwaveforms`) use to select between CPU (NumPy), CUDA (CuPy) and JAX implementations of native code at runtime. It is itself a hybrid Python / C++ / CUDA project built with `scikit-build-core` and CMake. Python ≥ 3.10 is required.

The package owns two responsibilities:
1. **Backend selection / loading machinery** — abstract `Backend`, `BackendsManager`, `Globals` singleton, the `ParallelModuleBase` class downstream code subclasses, and helpers (`get_backend`, `has_backend`, `pointer_adjust`, `wrapper`).
2. **Reference backend implementations** — a CPU and CUDA cubic-spline / interpolation kernel exposed as the concrete `GBT*Backend` classes used by the test suite and as a template for downstream packages, plus the pure-JAX `GBTJaxBackend`.

## Build & Install

The project uses `scikit-build-core` (configured in `pyproject.toml`) and CMake (`CMakeLists.txt`) — **do not use `setup.py` directly**.

```sh
# Editable dev install (CPU; auto-detects CUDA toolkit if present)
pip install -e '.[testing]'

# From-source install (release)
pip install .
```

Build behavior is controlled by the CMake option `GBT_WITH_GPU` (`AUTO` / `ON` / `OFF` / `ONLY` / `BARE`). On `AUTO` (default), the GPU backend is built only if `nvcc` and the CUDA toolkit are found. CUDA architectures are controlled by `GBT_CUDA_ARCH` (default `native`). Override via `pip install --config-settings=cmake.define.GBT_WITH_GPU=ON .` etc.

LAPACKE is required for the CPU backend. `GBT_LAPACKE_DETECT_WITH` (`AUTO`/`CMAKE`/`PKGCONFIG`/`DISABLE`) chooses how to find a system install; `GBT_LAPACKE_FETCH` (`AUTO`/`ON`/`OFF`) controls whether sources are downloaded and built. `GBT_LAPACKE_EXTRA_LIBS` defaults to `gfortran` when linkable — set explicitly if linker complains about missing Fortran symbols.

The CUDA plugin wheels are `gpubackendtools-cuda11x`, `-cuda12x`, `-cuda13x` (built with `GBT_WITH_GPU=ONLY`; this repo's `.github/workflows/publish.yml` builds only the main sdist + wheel). Each ships a `gbt_backend_cudaXXx` Python module that the main package imports lazily, on the backend's first lookup. The CPU module `gbt_backend_cpu` is built into the main wheel.

## Testing

Tests live in `tests/` and use `unittest`:

```sh
# Run the full suite
python -m unittest discover

# Run a single test file / case / method
python -m unittest tests.test_cubic_spline
python -m unittest tests.test_cubic_spline.CubicSplineTest
python -m unittest tests.test_cubic_spline.CubicSplineTest.test_cubic_spline
```

The `[testing]` extras (`matplotlib`) must be installed.

## Lint / Format

Pre-commit is configured with `ruff` (lint + format), `cmake-format`/`cmake-lint`, `pyproject-fmt`, `mypy`, and `clang-format` (Google style for C/C++/CUDA).

```sh
pre-commit install         # one-time
pre-commit run --all-files # run on the whole tree
```

`ruff` is on a **white-list** approach (see `pyproject.toml [tool.ruff]`) — most files are excluded; the white-list names `files/manager.py`, `files/registry.py`, `utils/config.py`, `utils/exceptions.py` and `utils/globals.py` under `src/gpubackendtools/`, of which only `utils/config.py` exists in the tree. `T201` ("no `print`") is enforced where ruff runs.

## Backend Architecture

This is the most important concept in the codebase — the design downstream packages mirror.

A **backend** = `(xp, native_methods)` where `xp` is `numpy`, `cupy` or `jax.numpy` and `native_methods` is a dataclass of C++/CUDA callables loaded from a per-flavor plugin module (`gbt_backend_cpu`, `gbt_backend_cuda11x`, `gbt_backend_cuda12x`, `gbt_backend_cuda13x`; the JAX backend has no plugin module and uses pure-Python functions). Each backend class sets two names: `_name`, the registry key `<pkg>_<flavor>` (`gbt_cpu`; `Backend.name`, used by the lookups and pickling), and `_backend_name`, the plugin module `<pkg>_backend_<flavor>` (`gbt_backend_cpu`; `Backend.backend_name`).

- **Class hierarchy** (`src/gpubackendtools/gpubackendtools.py`):
  - `Backend` (abstract) → `CpuBackend`, `Cuda11xBackend`, `Cuda12xBackend`, `Cuda13xBackend`. The CUDA classes share `_CudaBackend`, whose `check_cuda_backend` checks the plugin module and CuPy, and only when the first load fails checks the driver CUDA version and (on Linux) preloads the `nvidia-*` shared libs (`libcudart`, `libcublas`, `libnvJitLink`, `libcusparse`, `libnvrtc`) from the system search path or the `nvidia` Python wheels. `Cuda11xBackend.__init__` passes its own abstract loader instead of the subclass override, so any `*_cuda11x` backend raises `NotImplementedError` once those checks pass (see its docstring).
  - `BackendMethods` (dataclass) — the contract for what a backend exposes; just `xp` at the base.
  - `BackendsManager` — a registry mapping name → `BackendStatus{Unloaded, Loaded, Disabled, Unavailable}`. Lazily constructs backends on first access and caches the result. Every `add_backends` call rebuilds the registry for all packages (earlier instances are rebuilt on their next lookup), and `Globals.reset()` leaves an empty registry until the next `add_backends` call. The registry marks a backend unavailable only for `gpubackendtools.gpubackendtools.BackendUnavailableException` (and subclasses); the separate class of the same name in `gpubackendtools.exceptions`, which the `cutils` loaders raise, propagates out of the lookups.
- **Globals** (`globals.py`) — `Globals()` is a singleton holding the `BackendsManager`, logger, and `Configuration`. `get_backend(name)` / `has_backend(name)` / `get_first_backend([...])` are module-level shortcuts onto it.
- **Name aliasing**: only `get_backend` resolves aliases: `"<pkg>_cuda"` → the first available of `<pkg>_cuda13x`/`_cuda12x`/`_cuda11x`, and `"<pkg>_gpu"` aliases `"<pkg>_cuda"`. A bare flavor (`"cpu"`) is not a registry key and raises `ValueError`. `has_backend` takes exact registry keys only: the `cuda`/`gpu` aliases, bare or prefixed, raise `ValueError`. `get_first_backend` accepts full keys or bare flavors; a bare flavor matches the FIRST registered key whose part after the LAST `_` equals it (e.g. `cuda12x` → `gbt_cuda12x`). `gbt_*` registers first, so bare flavors resolve to GBT's backends, not to the calling package's (`lisatools_*`, `gbgpu_*`, ...): downstream code passes prefixed names.

### Adding a new backend method

To expose a new C++/CUDA function on the GBT backend:

1. Implement it in `src/gpubackendtools/cutils/Interpolate.cu` (or a new `.cu`/`.hh` pair) and bind it in `gbt_binding.cxx` using nanobind.
2. Add a field to `GBTBackendMethods` in `src/gpubackendtools/cutils/__init__.py`.
3. Wire the field into `GBTBackend.__init__` and into every loader so it is loaded from the right plugin module: `GBTCpuBackend.cpu_methods_loader()`, `GBT{Cuda11x,Cuda12x,Cuda13x}Backend.cuda*_module_loader()`, and `jax/backend.py::_jax_methods_loader()` (the `GBTBackendMethods` fields have no defaults, so a loader that does not fill the new field fails).

### Subclassing for downstream packages

Downstream packages register backends by subclassing `BackendMethods` (one field per native symbol), writing a mixin that copies those fields onto the backend, deriving one class per flavor from the device base (`CpuBackend`, `Cuda12xBackend`, ...) plus the mixin, and adding them to the global manager (full contract and example: `docs/source/user/gpubackendtools.rst`, "Writing a downstream backend"):

```python
from gpubackendtools import Globals
Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend, ...})
```

User-facing classes inherit from `ParallelModuleBase` (`parallelbase.py`), which exposes `self.backend`, `self.xp`, `self.backend_name`, and `build_with_same_backend(...)`. Subclasses must implement `supported_backends()` (use the `CPU_ONLY`, `GPU_ONLY`, `GPU_RECOMMENDED`, `CPU_RECOMMENDED_WITH_GPU_SUPPORT` static helpers). Return PREFIXED names from `supported_backends()` (`["mypkg_" + f for f in cls.GPU_RECOMMENDED()]`); the helpers return bare flavors, which `get_first_backend` resolves to `gbt_*`. A plain `ParallelModuleBase` passes a string `force_backend` to `get_backend`, so it needs a full registry name. `GBTParallelModuleBase` is the GBT-namespaced variant that prefixes `gbt_` onto a string `force_backend` not already starting with it; downstream bases copy the pattern with their own prefix.

## Native Sources (`src/gpubackendtools/cutils/`)

- `Interpolate.cu` / `Interpolate.hh` — heavy spline build/solve (LAPACKE tridiagonal on CPU, cuSPARSE on GPU, plus PCR launcher on GPU). The `.cu` is **copied to `Interpolate.cxx` at build time** and compiled by the C++ compiler for the CPU backend; the same `.cu` is compiled by `nvcc` for the GPU backend. **Code must be valid as both** — guard CUDA-only intrinsics with the macros in `gbt_global.h`.
- **`InterpolateDevice.hh`** (Phase 1, 2026-06-02) — header-only `__device__` cubic-spline evaluators (`CubicSpline`, `CubicSplineSegment`, `get_window`, `binary_search`, `even_sampled_search`, `eval_single`, `eval`). Downstream `.cu` files `#include "InterpolateDevice.hh"` to evaluate splines without linking against `Interpolate.cu`. All method bodies that used to be out-of-line in `Interpolate.cu` are now inlined here.
- `gbt_binding.cxx` / `gbt_binding.hpp` — **nanobind** module (`NB_MODULE`) exposing the C++/CUDA functions to Python (one shared binding source for both CPU and GPU builds; the produced extension is `gbt_backend_<flavor>.interp`). Migrated from pybind11 at Phase 3M (2026-06-04); the bespoke `pybind11_cuda_array_interface.hpp` CUDA-array caster was retired in the same migration — nanobind's `nb::ndarray<T, nb::device::cuda>` covers that case natively.
- `cuda_complex.hpp` — host/device-portable complex type. **Single copy shared across LISA Analysis Tools**; the LAT, GBGPU and BBHx local duplicates were deleted (LAT at Phase 2d); FEW keeps its own diverged copy.
- **`GPUBackendToolsConfig.cmake`** (Phase 1) — forward-compat CMake config for downstream `find_package(GPUBackendTools CONFIG REQUIRED)` → `GPUBackendTools::headers` interface target.
- `CubicSpline.cu`, `interp.pyx` — legacy Cython artifacts (not built by CMake). The active path is the nanobind module produced from `gbt_binding.cxx`. Cython sources are excluded from wheels (`wheel.exclude` in `pyproject.toml`).
- `cmake_functions.cmake` — `apply_cpu_backend_common_options` / `apply_gpu_backend_common_options` helpers and `get_lapacke()` LAPACKE detector.

## Downstream consumption (Phase 1)

Downstream native builds (LAT, GBGPU, BBHx) consume GBT's `cutils/` directory
(headers, `Interpolate.cu`, `cmake_functions.cmake`); FastEMRIWaveforms' CMake does not
reference GBT. GBT exposes the directory two ways:
- `gpubackendtools.get_include() -> str` — absolute path to `cutils/`. The LAT, GBGPU and BBHx `cutils/CMakeLists.txt` compute the same directory with their own `execute_process` Python shell-out (`gpubackendtools.__file__` directory + `cutils`, stored as `GBT_CUTILS`) rather than calling `get_include()`, then `include(${GBT_CUTILS}/cmake_functions.cmake)` (via `GBT_CMAKE_TOOLS`) and add `GBT_CUTILS` to the include path.
- `gpubackendtools.get_cmake_module_path() -> str` — same path, packaged with `GPUBackendToolsConfig.cmake` for `find_package(GPUBackendTools CONFIG)` callers.

## Python package layout (`src/gpubackendtools/`)

- `__init__.py` — registers the four `GBT*Backend` classes (plus `gbt_jax` when JAX imports) on `Globals().backends_manager` at import. Its `__all__` lists `get_logger` / `get_config` / `get_config_setter` / `get_file_manager`, which the package namespace does not define, so `from gpubackendtools import *` raises `AttributeError`.
- `gpubackendtools.py` — `Backend`, `BackendMethods`, `BackendsManager`, all the CUDA-version-checking and `nvidia-*` dynlib-loading logic, plus `BackendStatus*` records.
- `globals.py` — `Globals` singleton, `Configuration` plumbing, and the `get_backend` / `has_backend` / `get_first_backend` / `initialize` / `reset` module-level API. Its module-level `add_backends` raises `AttributeError` (it calls a `Globals.add_backends` that does not exist); register with `Globals().backends_manager.add_backends(...)`.
- `parallelbase.py` — `ParallelModuleBase` (the base class downstream "compute object" classes inherit from) and `GBTParallelModuleBase` (the GBT-flavored namespace variant).
- `interpolate.py` — `CubicSplineInterpolant`, the reference user-facing class wrapping the native `interpolate_wrap` / `CubicSplineWrap` / `CubicSpline` exposed via `GBTBackendMethods`.
- `pointeradjust.py` — `wrapper(...)` and `pointer_adjust` decorator: convert NumPy/CuPy arrays and Cython-class objects to raw `size_t` pointers for legacy Cython entry points. No live caller: `interpolate.py` imports `wrapper` but never calls it, and the legacy `cutils/interp.pyx` that uses it is not built.
- `cutils/__init__.py` — concrete `GBTCpuBackend` / `GBTCuda11xBackend` / `GBTCuda12xBackend` / `GBTCuda13xBackend`, the `GBTBackend` mixin and the `GBTBackendMethods` dataclass.
- `jax/` — the JAX backend: `GBTJaxBackend` (`_name = "gbt_jax"`, `backend.py`), with a scipy spline fit on the host and pure-JAX evaluation (`cubic_spline.py`). No compiled plugin module — gated on `import jax` only, subclasses `Backend` directly (skips the native-module-installed check). Declares only `Feature.NUMPY` (`uses_gpu` is `False`) and its first load turns on `jax_enable_x64` process-wide. Selected with `get_backend("gbt_jax")` or `force_backend="jax"` on a `GBTParallelModuleBase` subclass. Documented in the docstrings and on the Sphinx API pages.
- `exceptions.py` — `GPUBACKENDTOOLSException` root, `BackendUnavailableException`, `BackendNotInstalled`, `CudaException`, `CuPyException`, `MissingDependency`; re-exports `ExceptionGroup`. `gpubackendtools.py` defines its OWN `BackendUnavailableException` / `BackendNotInstalled` plus `MissingDependencies` / `MissingDriver` / `SoftwareException` / `MissingHardware` / `BackendAccessException`. The registry catches only the `gpubackendtools.gpubackendtools` classes, so a loader must raise those (the in-tree `cutils` loaders raise the `.exceptions` one, which escapes the lookups).
- `utils/` — `config.py` (Configuration / ConfigConsumer / `detect_cfg_file` for CLI/env/file-based config, prefix `GPUBACKENDTOOLS_`); `citation.py`, `citations.py` and `utility.py` have no importer (TODO-tagged), and `citations.py` cannot be imported (it imports a missing `gpubackendtools.utils.exceptions`).

## Key External Dependencies

- `nanobind` (pinned exactly, `==2.12.0` — a LISA Analysis Tools–wide constraint for cross-wheel type-sharing ABI compatibility), `cython`, `numpy`, `scikit-build-core` — build-time.
- `nvidia-ml-py` (`pynvml`) — runtime CUDA driver-version detection in `_CudaBackend._get_cuda_version`.
- `pydantic`, `jsonschema`, `pyyaml` — declared dependencies, imported only by the dead `utils/citations.py` (the file/registry layer they were declared for is gone). `platformdirs` is used by `utils/config.py::detect_cfg_file`.
- `cupy` is **not** declared in `dependencies` — it must be installed separately matching the chosen CUDA backend (`cupy-cuda12x` etc.). The CUDA backend constructors raise `MissingDependencies` listing the right `cupy-cudaXXx` and `nvidia-*-cuXX` pip packages when something is missing.
- `lapacke` (and `gfortran`) — system requirement for the CPU backend, fetched and built if not found (controlled by `GBT_LAPACKE_FETCH`).

## Notes for Working in This Tree

- The **`.cu` → `.cxx` copy** rebuild step is implicit: editing `Interpolate.cu` triggers rebuilds of both CPU and GPU targets. Do not edit the generated `Interpolate.cxx` directly.
- When adding Python code that should work on both CPU and GPU, take an `xp` (or use `self.xp` from `ParallelModuleBase`) rather than importing `cupy` unconditionally. The module-level `try: import cupy` / `except ImportError` fallback in `pointeradjust.py` (which sets a `gpu` flag) is acceptable for module-load-time gating but inside class methods prefer routing through the backend.
- `pyproject.toml` contains placeholder tokens (`#@NAMESUFFIX@`, `#@SKIP_PLUGIN@`, `#@DEPS_*@`, `#@FALLBACK_VERSION@`, `#@ADD_CUDA_HERE` in CMakeLists.txt) consumed by the plugin-wheel build pipeline (`gpubackendtools-cuda{11,12,13}x`). Don't remove them when editing those files.
- `dist/` (gitignored; absent in the current checkout) holds wheels from local builds when present — not source of truth.
- `docs/_build/` (Sphinx output) and `src/gpubackendtools/_version.py` (auto-generated by `setuptools_scm`, untracked) are dev artifacts — leave them alone unless specifically asked.
- Build with `pip` / `cmake` only — there is no Makefile-based build workflow (`docs/Makefile` is only the Sphinx docs makefile).

## Backend implementation hierarchy (LISA Analysis Tools–wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START_X`,
   `BLOCK_INCR_X`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.

