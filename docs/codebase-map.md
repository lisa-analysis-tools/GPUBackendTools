_Last mapped: e42f469 · 2026-07-10; statements re-verified against `src/gpubackendtools/` at a63f363 · 2026-10-19 (docs audit) · regenerate when structure changes_

> Repo: `GPUBackendTools` (package import name `gpubackendtools`, current git branch `spline`).
> Role: **foundation library** — owns the compute-backend registry and shared C++/CUDA
> plumbing that LISAanalysistools (LAT), GBGPU, BBHx, FastEMRIWaveforms (FEW) and Eryn all sit on.

## 1. What this is

`gpubackendtools` is a hybrid Python/C++/CUDA package (built with `scikit-build-core` + CMake)
that provides (a) a generic **backend-selection framework** — abstract `Backend` classes, a
`BackendsManager` registry, a `Globals()` singleton, and a `ParallelModuleBase` base class — that lets
any "compute object" pick CPU (NumPy), CUDA 11/12/13x (CuPy), or JAX at construction time and
dispatch consistently thereafter; and (b) a **reference implementation** of that framework: a
cubic-spline / interpolation kernel (`CubicSplineInterpolant` + native `Interpolate.{cu,hh}` +
nanobind bindings) that doubles as the template every downstream package copies to add its own
backends. It also ships the C++ headers shared across LISA Analysis Tools (`cuda_complex.hpp`, `gbt_global.h`,
`InterpolateDevice.hh`) that downstream native code `#include`s directly.

## 2. Layout

- `src/gpubackendtools/__init__.py` — registers the 4 GBT backends (`gbt_cpu`, `gbt_cuda{11,12,13}x`) + optional `gbt_jax`; exposes `get_include()` / `get_cmake_module_path()`.
- `src/gpubackendtools/gpubackendtools.py` — `Backend`, `BackendMethods`, `CpuBackend`, `Cuda{11,12,13}xBackend`, `_CudaBackend`, `BackendsManager`, `BackendStatus*`. The abstract machinery.
- `src/gpubackendtools/globals.py` — `Globals` singleton (config + logger + `BackendsManager`), `ConfigurationSetter`; module-level `get_backend`/`has_backend`/`get_first_backend`, `get_logger`/`get_config`/`get_config_setter`/`initialize`/`reset`. The module-level `add_backends` raises `AttributeError` (it calls a `Globals.add_backends` that does not exist; TODO-tagged): register with `Globals().backends_manager.add_backends(...)`.
- `src/gpubackendtools/parallelbase.py` — `ParallelModuleBase` (generic) and `GBTParallelModuleBase` (prefixes `force_backend` with `"gbt_"`). The class downstream "compute object" classes subclass.
- `src/gpubackendtools/interpolate.py` — `CubicSplineInterpolant`, the reference `ParallelModuleBase` subclass; user-facing spline wrapper around the native backend.
- `src/gpubackendtools/pointeradjust.py` — `wrapper()`/`pointer_adjust` — legacy Cython pointer-marshalling helpers (numpy/cupy array → raw `size_t`).
- `src/gpubackendtools/cutils/` — native sources: `Interpolate.{cu,hh}`, `InterpolateDevice.hh`, `gbt_binding.{cxx,hpp}` (nanobind), `cuda_complex.hpp`, `gbt_global.h`, `cmake_functions.cmake` (LAPACKE detection + the `apply_{cpu,gpu}_backend_common_options` helpers that downstream CMake also includes), `GPUBackendToolsConfig.cmake`, `CMakeLists.txt`; legacy `CubicSpline.cu` / `interp.pyx` (not built). Also `cutils/__init__.py` — concrete `GBTCpuBackend`/`GBTCuda{11,12,13}xBackend`, the `GBTBackend` mixin and the `GBTBackendMethods` dataclass.
- `src/gpubackendtools/jax/` — `backend.py` (`GBTJaxBackend`), `cubic_spline.py` (scipy fit on the host + pure-JAX evaluation), gated on `import jax`.
- `src/gpubackendtools/utils/` — `config.py` (`Configuration`/`InitialConfigConsumer`/`ConfigEntry`/env-var+CLI+INI config plumbing, prefix `GPUBACKENDTOOLS_`); `citation.py`, `citations.py` and `utility.py` (`cuda_set_device`) have no importer (TODO-tagged), and `citations.py` is not even importable (it imports a missing `gpubackendtools.utils.exceptions`).
- `src/gpubackendtools/exceptions.py` — `GPUBACKENDTOOLSException` root + `BackendUnavailableException`, `BackendNotInstalled`, `CudaException`, `CuPyException`, `MissingDependency`; re-exports `ExceptionGroup`. `gpubackendtools.py` declares its OWN `BackendUnavailableException` / `BackendNotInstalled` plus `MissingDependencies` / `MissingDriver` / `SoftwareException` / `MissingHardware`; the registry catches only the `gpubackendtools.gpubackendtools` classes (see §7).
- `CMakeLists.txt` (root) + `cmake/CPM.cmake` — top-level CMake project; `GBT_WITH_GPU` (AUTO/ON/OFF/ONLY/BARE), `GBT_CUDA_ARCH`, `GBT_MARCH`, LAPACKE knobs (`GBT_LAPACKE_DETECT_WITH`, `GBT_LAPACKE_FETCH`, `GBT_LAPACKE_EXTRA_LIBS`).
- `tests/test_cubic_spline.py` — the only test file: `CubicSplineTest` (`CubicSplineInterpolant` and the native `cpp_class.eval_wrap` against `scipy.interpolate.CubicSpline`) and `SingleSplineFitTest` (`fit_cubic_spline_thomas` on `gbt_cpu`, `fit_cubic_spline_pcr` on a GPU backend when CuPy imports).
- `docs/source/` (Sphinx: `index.rst` + API pages `user/{gpubackendtools,parallelbase,interpolate,globals}.rst`; `conf.py` copies the tutorial notebook in at build time), `examples/GPUBackendTools_tutorial.ipynb` — tutorial notebook.

## 3. Core abstractions

```
Globals()  (singleton: logger, Configuration, BackendsManager)
   └── backends_manager: BackendsManager
          ├── _known_backends: {name -> Backend subclass}   (populated by add_backends())
          └── _registry: {name -> BackendStatus}             (Unloaded/Loaded/Disabled/Unavailable, lazy)

Backend (abstract)                     BackendMethods (dataclass: xp)
 ├── CpuBackend            (abstract)  → cpu_methods_loader()
 ├── _CudaBackend                       → check_cuda_backend() [module + cupy + driver-version checks]
 │    ├── Cuda11xBackend   (abstract)  → cuda11x_module_loader() / cuda11x_dynlib_loader()
 │    ├── Cuda12xBackend   (abstract)  → cuda12x_module_loader() / cuda12x_dynlib_loader()
 │    └── Cuda13xBackend   (abstract)  → cuda13x_module_loader() / cuda13x_dynlib_loader()
 └── (JAX subclasses Backend directly, not CpuBackend — no importlib module check)

Concrete GBT backends (src/gpubackendtools/cutils/__init__.py, jax/backend.py):
  GBTCpuBackend    _name="gbt_cpu"     _backend_name="gbt_backend_cpu"    xp=numpy
  GBTCuda11xBackend _name="gbt_cuda11x" _backend_name="gbt_backend_cuda11x" xp=cupy
  GBTCuda12xBackend _name="gbt_cuda12x" _backend_name="gbt_backend_cuda12x" xp=cupy
  GBTCuda13xBackend _name="gbt_cuda13x" _backend_name="gbt_backend_cuda13x" xp=cupy
  GBTJaxBackend     _name="gbt_jax"     _backend_name="gbt_backend_jax"     xp=jax.numpy (no compiled plugin — pure-Python)

ParallelModuleBase  (src/gpubackendtools/parallelbase.py)
  __init__(force_backend=None|str|Backend|(pkg,flavor))
  .backend -> Backend        .xp -> Backend.xp        .backend_name -> str
  .supported_backends()  [abstract; use CPU_ONLY/GPU_ONLY/GPU_RECOMMENDED/CPU_RECOMMENDED_WITH_GPU_SUPPORT]
  .build_with_same_backend(cls, args, kwargs)  /  .adapt_backend_kwargs(kwargs)
GBTParallelModuleBase(ParallelModuleBase)   # prefixes "gbt_" onto any force_backend string not starting with it
```

- **`Backend.name`** is the *registry key* (`"gbt_cpu"`, `"gbt_cuda12x"`, ...) — what
  `get_backend()`/`BackendsManager` index by. **`Backend.backend_name`** is the *plugin module
  name* (`"gbt_backend_cpu"`) — what `_check_module_installed` imports. Don't confuse the two;
  `Backend.__reduce__` pickles by `.name`, not `.backend_name`.
- `BackendsManager.get_first_backend([...])` accepts full keys or bare flavors. A bare flavor
  matches the **first registered key whose part after the LAST `_` equals it**
  (`name.split("_")[-1]`), in registration order. `gbt_*` is registered first (at
  `import gpubackendtools`), so a bare list such as `GPU_RECOMMENDED()` resolves to
  `gbt_*` backends, NOT to the calling package's own: downstream `supported_backends()` must
  return prefixed names (`["mypkg_" + f for f in cls.GPU_RECOMMENDED()]`).
- Only `globals.get_backend` resolves the aliases: `"<pkg>_cuda"` → first available of
  `<pkg>_cuda13x → _cuda12x → _cuda11x`, and `"<pkg>_gpu"` → same. `has_backend` takes exact
  registry keys only (`has_backend("gbt_cuda")` raises `ValueError`; its bare `"cuda"`/`"gpu"`
  special case recurses into unregistered bare names and raises `ValueError` too).
  `BackendsManager.get_backend` / `.has_backend` take exact keys.
- `ParallelModuleBase(force_backend=...)` hands a string to `globals.get_backend`, so a bare
  flavor (`"cpu"`) raises `ValueError` there; bare flavors work only through a subclass that
  adds its package prefix (`GBTParallelModuleBase`, `LISAToolsParallelModule`,
  `GBGPUParallelModule`, `BBHxParallelModule`, `FewBackendConsumer`).
- **Registry resets.** Every `BackendsManager.add_backends(...)` call rebuilds `_registry`
  from `_known_backends` (all packages), so instances loaded earlier are rebuilt on their next
  lookup and the `get_backend(x) is get_backend(x)` identity does not survive a later
  registration. `Globals.reset()` / `get_config_setter(reset=True)` leave a new, EMPTY
  registry until the next `add_backends` call (`add_backends({})` re-publishes the known
  classes).

**Cubic-spline interpolation** (the reference backend implementation):
- `interpolate.py::CubicSplineInterpolant(GBTParallelModuleBase)` — builds `x/y/c1/c2/c3` flat
  arrays, dispatches spline *fit* to `self.backend.interpolate_wrap` (native) or a functional JAX
  path (`self.backend.name == "gbt_jax"` branch — C++ backends mutate buffers in place; JAX
  can't, so it returns coefficients instead), then evaluates splines in pure Python/NumPy/CuPy in
  `__call__` (`searchsorted2d_vec` segment lookup + cubic eval — this evaluation path does *not*
  call into the C++ `CubicSpline` class; it's a parallel NumPy-side implementation; the native
  evaluator is reachable through `cpp_class.eval_wrap`). Out-of-range queries raise by
  default; with `error_out_of_bounds=False` they warn and return `0.0` (zeros, not
  clamping). On `gbt_jax` the in-place index fixes in `__call__` (query at the last knot,
  zeroing) fail on immutable JAX arrays; `jax.cubic_spline.evaluate_cubic_spline` instead
  clips the segment index, i.e. extrapolates with the end cubics.
- Spacing codes `CUBIC_SPLINE_{LINEAR,LOG10,GENERAL}_SPACING` = 1/2/3 (`interpolate.py`,
  `InterpolateDevice.hh`, and the `interp` module attributes) are auto-detected from `x` and
  stored as `spline_type`; the `spline_type` constructor argument is ignored.
- `cutils/InterpolateDevice.hh` — **header-only** `CubicSpline`/`CubicSplineSegment` device
  classes (`eval`, `eval_single`, `get_window`, `binary_search`, `even_sampled_search`,
  derivatives). Downstream `.cu` files `#include` this directly to evaluate a spline without
  linking GBT's `Interpolate.cu` TU. Class name is aliased `CubicSpline` → `CubicSplineGPU` /
  `CubicSplineCPU` via `#define` at the top of the header (see §7 — CPU/GPU aliasing rule).
- `cutils/Interpolate.{cu,hh}` — the *build/solve* side: `interpolate()` (batched tridiagonal
  solve: LAPACKE `dgtsv` on CPU, cuSPARSE `cusparseDgtsv2StridedBatch` on GPU),
  `fit_cubic_spline_thomas` (CPU-only Thomas algorithm), `fit_cubic_spline_pcr` (GPU-only
  Parallel Cyclic Reduction, one block/spline, shared-memory scratch). For the CPU target
  `Interpolate.cu` is **copied to `Interpolate.cxx`** at CMake time and compiled by the C++
  compiler; the GPU target compiles `Interpolate.cu` with nvcc — do not edit the generated
  `.cxx`.
- `cutils/gbt_binding.{cxx,hpp}` — **nanobind** module (`NB_MODULE(interp, m)`) exposing
  `CubicSplineWrap` (aliased `CubicSplineWrapGPU`/`CubicSplineWrapCPU`), `CubicSpline`
  (`CubicSplineGPU`/`CubicSplineCPU`), `interpolate_wrap`, `fit_cubic_spline_thomas` (CPU build
  only) / `fit_cubic_spline_pcr` (CUDA build only), and the `CUBIC_SPLINE_*` constants. One
  binding source compiles for both flavors via `#if defined(__CUDA_COMPILATION__)`. This is the
  **sole registrant** of `CubicSplineWrap`/`CubicSpline` — no other repo re-registers these types.
- `cutils/cuda_complex.hpp` — host/device-portable complex type; **single source shared across
  LISA Analysis Tools** (GBGPU/BBHx local duplicates were deleted at Phase 3.dedup; FEW's diverged
  copy is intentionally left alone — see MEMORY).
- `cutils/gbt_global.h` — `CUDA_CALLABLE_MEMBER`/`CUDA_DEVICE`/`CUDA_KERNEL`/`CUDA_SHARED`,
  per-axis `THREAD_START_{X,Y,Z}`/`BLOCK_INCR_{X,Y,Z}`/`BLOCK_START_{X,Y,Z}`/`GRID_INCR_{X,Y,Z}`
  macros (GPU: real CUDA intrinsics; CPU: degenerate to a single-iteration loop), `gpuErrchk`,
  and `typedef gcmplx::complex<double> cmplx`. Included transitively by nearly every kernel header
  across the LISA Analysis Tools repos.

## 4. Public API / entry points

Top-level names of `gpubackendtools/__init__.py` (downstream packages also import from the
submodules: `gpubackendtools.gpubackendtools` for `Backend` / `BackendMethods` /
`CpuBackend` / `Cuda{11,12,13}xBackend`, `gpubackendtools.exceptions` (star-imported by every
`cutils`), `gpubackendtools.globals`, `gpubackendtools.parallelbase`,
`gpubackendtools.interpolate`):

- `get_backend(name)` / `has_backend(name)` / `get_first_backend([...])` — the
  `gpubackendtools.globals` functions, shortcuts onto `Globals().backends_manager` (alias
  rules in §3).
- `Globals()` — the singleton itself (`Globals().backends_manager.add_backends({...})` is how a
  downstream package registers its own backend classes at import time).
- `get_include() -> str` — absolute path to `cutils/` for `#include "gbt_global.h"` /
  `"cuda_complex.hpp"` / `"InterpolateDevice.hh"` and for `cmake_functions.cmake`. The
  downstream CMake files (LAT, GBGPU, BBHx) compute the same directory with their own Python
  shell-out (`gpubackendtools.__file__` directory + `cutils`) rather than calling
  `get_include()`.
- `get_cmake_module_path() -> str` — same directory, for `find_package(GPUBackendTools CONFIG)` →
  `GPUBackendTools::headers` interface target (via bundled `GPUBackendToolsConfig.cmake`); header-
  only, exports no compiled targets (every downstream recompiles its own TUs).
- `ParallelModuleBase` — the base class downstream "compute object" classes subclass
  (`GBTParallelModuleBase` is not re-exported: `gpubackendtools.parallelbase`).
- `wrapper()` / `pointer_adjust` — legacy Cython pointer-marshalling helpers (the active
  native path is nanobind, not these). No live caller: `interpolate.py` imports `wrapper` but
  never calls it, LAT's `response/tdionfly.py` calls it only in branches that first raise
  `NotImplementedError`, and the legacy `cutils/interp.pyx` that uses it is not built.
- `interpolate.CubicSplineInterpolant` — concrete reference class (not re-exported at top level;
  `from gpubackendtools.interpolate import CubicSplineInterpolant`).
- `__all__` also lists `get_logger`, `get_config`, `get_config_setter`, `get_file_manager`,
  which the package namespace does not define, so `from gpubackendtools import *` raises
  `AttributeError` (import them from `gpubackendtools.globals`; `get_file_manager` does not
  exist).
- Sphinx API pages: `docs/source/user/gpubackendtools.rst` (names, lookup rules, the
  downstream-backend contract, backend classes, registry, GBT/JAX backends, exceptions),
  `parallelbase.rst`, `interpolate.rst` (spline + JAX spline functions), `globals.rst`
  (globals, env-var configuration).

## 5. Backend-wheel model

- The **CPU backend is built into the main wheel** (`gpubackendtools`, default
  `GBT_WITH_GPU=AUTO`, which also builds the CUDA backend when a CUDA toolkit is found); the
  `GBT_WITH_GPU=ONLY` mode (CPU off, GPU on) is the plugin-wheel build, and the CUDA loaders'
  pip hint names the plugin packages `gpubackendtools-cuda11x` / `-cuda12x` / `-cuda13x`. This
  repo's `.github/workflows/publish.yml` builds only the main sdist + wheel. Every compiled
  nanobind extension is installed under a distinctly-named import package: `gbt_backend_cpu`,
  `gbt_backend_cuda11x`, `gbt_backend_cuda12x`, `gbt_backend_cuda13x` (target dir computed in
  `src/gpubackendtools/cutils/cmake_functions.cmake::apply_{cpu,gpu}_backend_common_options`
  as `${pkg_name}_backend_cpu` / `${pkg_name}_backend_cuda${CUDAToolkit_VERSION_MAJOR}x`, with
  `pkg_name=gbt`). Backends are constructed on first lookup: the CUDA bases check the plugin
  module, CuPy and (on a failed first load) the driver version inside `check_cuda_backend()`,
  whose `module_loader` (`GBTCuda12xBackend.cuda12x_module_loader`, etc.) imports
  `gbt_backend_cuda12x.interp` — so a machine can have 0, 1, or all 3 CUDA plugins installed
  and only the importable ones become `BackendStatusLoaded`. Caveat: `Cuda11xBackend.__init__`
  passes the abstract `Cuda11xBackend.cuda11x_module_loader` (not the subclass override), so
  any `*_cuda11x` backend raises `NotImplementedError` once its plugin and CuPy checks pass.
- **`Backend.__reduce__`** (with the module-level `_resolve_backend_for_unpickle`, both in
  `gpubackendtools.py`) makes every `Backend` instance picklable
  *by registry name* rather than by value: pickle stores `(_resolve_backend_for_unpickle, (name,))`
  and unpickling re-looks-up the singleton via `get_backend(name)` (importing the owning package
  first if needed, since the registry-name prefix — `lisatools_`, `gbgpu_`, ... — doubles as the
  import name; `gbt_*` keys resolve because unpickling imports `gpubackendtools.gpubackendtools`,
  which runs `gpubackendtools/__init__.py`). In the same process `loads(dumps(b)) is b` unless a
  later `add_backends` call rebuilt the registry. `__deepcopy__`/`__copy__` both `return self`
  for the same reason: a `Backend` is a process-wide singleton (it holds a live `xp` module reference and compiled-extension callables,
  neither of which is a meaningful "value" to duplicate). Committed 2026-07-07 (`bd6355c`,
  `e42f469`) — this is what lets `Backend` instances live safely inside settings trees that get
  `deepcopy`/`pickle`d (the LISA Analysis Tools–wide deepcopy/pickle-safety rule, see §7).
- **The `xp` property pattern**: `ParallelModuleBase.xp` returns `self.backend.xp` — never cache
  `self.xp = cp`/`self._xp = ...` as a plain attribute (that reintroduces the unpicklable-module
  problem `Backend.__reduce__` was built to solve one layer up). `CubicSplineInterpolant.xp` is
  itself just `self.backend.xp`.
- **Co-existence**: `has_backend("gbt_cpu")` and `has_backend("gbt_cuda12x")` can both be `True`
  simultaneously in the same interpreter because they're registered under distinct keys in one
  shared `BackendsManager._registry`, each lazily loaded from a distinctly-named Python package
  with distinctly-named C++ types (see §7 CPU/GPU aliasing) — no symbol or `typeid` collision
  between the CPU and GPU `.so`s even though both may be imported in the same process.

## 6. Cross-repo role

Every other LISA Analysis Tools repo depends on GBT. Verified via grep across sibling repos:

| Consumer | Python import | CMake |
|---|---|---|
| LISAanalysistools | `lisatools/__init__.py`, `response/tdionfly.py`, `cutils/__init__.py`, `utils/parallelbase.py`, `jax/backend.py`, `sources/emri/chebyshevwave.py` | `cutils/CMakeLists.txt` computes `GBT_CUTILS` with a Python shell-out (`gpubackendtools.__file__` directory + `cutils`, the `get_include()` directory), includes `${GBT_CUTILS}/cmake_functions.cmake`, checks `${GBT_CUTILS}/gbt_global.h` exists, adds `GBT_CUTILS` to `target_include_directories` |
| GBGPU | `gbgpu.py`, `__init__.py`, `cutils/__init__.py`, `parallelbase.py` | `cutils/CMakeLists.txt` (same pattern, plus copy-compiles `${GBT_CUTILS}/Interpolate.cu`) |
| BBHx | `__init__.py`, `waveformbuild.py`, `cutils/__init__.py`, `utils/parallelbase.py` | `cutils/CMakeLists.txt` (same pattern, plus copy-compiles `${GBT_CUTILS}/Interpolate.cu`) |
| FastEMRIWaveforms | `cutils/__init__.py`, `__init__.py`, `utils/utility.py`, `utils/baseclasses.py` | no `GBT_CUTILS` / `gpubackendtools` use in its CMake; **not declared in `pyproject.toml` dependencies** despite being imported (possible doc/packaging gap, not fixed here) |
| Eryn | `paraensemble.py` | pulled in via `pyproject.toml` dependency (no native build) |

Mechanism: (1) **Python** — every downstream `ParallelModuleBase`-style base class
(`lisatools.utils.parallelbase.LISAToolsParallelModule`, `gbgpu.parallelbase.GBGPUParallelModule`,
`bbhx.utils.parallelbase.BBHxParallelModule`, `few.utils.baseclasses.FewBackendConsumer`)
subclasses GBT's `ParallelModuleBase`, adds its package prefix to bare `force_backend`
strings, and its package registers its own backends onto the *same*
`Globals().backends_manager` singleton via `add_backends({"<pkg>_cpu": ..., ...})`. (2) **CMake** —
the LAT, GBGPU and BBHx native builds locate GBT's `cutils/` directory (assigned to
`GBT_CUTILS`), include its `cmake_functions.cmake` helpers, add it to
`target_include_directories`, then `#include` `gbt_global.h`/`cuda_complex.hpp`/
`InterpolateDevice.hh` directly; **no compiled GBT library is linked** — GBGPU and BBHx
copy-compile `Interpolate.cu` into their own extensions, and Python-side consumers use GBT's
importable `gbt_backend_<flavor>.interp` module (the LAT/GBGPU/BBHx loaders import it for
`CubicSplineWrap`).

## 7. Non-obvious invariants / gotchas

- **Single-registrant rule (L2, LISA Analysis Tools–wide)**: `CubicSplineWrap`/`CubicSpline` (and by extension
  any nanobind-registered type) must be registered by exactly one binding TU per backend flavor.
  GBT's `gbt_binding.cxx` is the sole registrant for these two types; the grep gate shared across
  LISA Analysis Tools (`LISAanalysistools/tools/check_single_registrant.sh`, not in this repo)
  enforces it elsewhere.
- **Two `BackendUnavailableException` classes.** `gpubackendtools.gpubackendtools` and
  `gpubackendtools.exceptions` each define one (siblings, neither derives from the other). The
  registry (`_try_loading_backend`) and `check_cuda_backend`'s retry steps catch only the
  `gpubackendtools.gpubackendtools` class, but `cutils/__init__.py` star-imports
  `gpubackendtools.exceptions` and its loaders raise THAT class (LAT/GBGPU/BBHx/FEW `cutils`
  copy the pattern). So when a plugin package imports but its `interp` extension does not, the
  exception escapes `get_backend` / `has_backend` / `get_first_backend` instead of marking the
  backend unavailable. A missing plugin package is still handled, because the device bases
  check the plugin module first (raising the `gpubackendtools.gpubackendtools` classes).
- **CPU/GPU class-name aliasing**: every type that ends up in both the CPU and GPU `.so` (wrapper
  classes *and* the underlying C++ classes) must be `#define`d to a CPU/GPU-suffixed alias in both
  branches of an `#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)` block — see
  `gbt_binding.hpp:39-43` (`CubicSplineWrap`) and `InterpolateDevice.hh:13-17` (`CubicSpline`).
  Missing this on one branch produces a backend-asymmetric class name and silent `typeid`
  collisions between the two plugin wheels.
- **`Interpolate.cu` → `Interpolate.cxx` copy is implicit** (the `add_custom_command` at the
  top of the CPU block of `cutils/CMakeLists.txt`): editing `Interpolate.cu` triggers rebuilds
  of *both* CPU and GPU targets; never edit the generated `.cxx` directly.
- **`nanobind==2.12.0` pinned exactly** in `pyproject.toml` — "cross-wheel nanobind type sharing
  (OrbitsWrap, CubicSplineWrap, ...) requires every separately-built wheel to embed the same
  nanobind internals ABI. Bump in ONE sprint-wide PR only." The project migrated pybind11 →
  nanobind at Phase 3M; `gbt_binding.cxx`/`.hpp` are nanobind (`#include <nanobind/...>`,
  `NB_MODULE`). (Some docstrings/comments, e.g. in `tests/test_cubic_spline.py` and the
  tutorial notebook, still say "pybind11".)
- **Deepcopy/pickle safety**: don't cache `self.xp = cp` as a plain attribute anywhere in an
  object that might enter a settings tree; use the `xp` *property* pattern (§5). `Backend`
  instances are deepcopy/pickle-safe via `__reduce__`/`__deepcopy__`/`__copy__` as of 2026-07-07
  (`bd6355c`, `e42f469`) — this repo is the origin of that LISA Analysis Tools–wide rule.
- **`cuda_complex.hpp` sole-source**: GBT is the single source shared across LISA Analysis Tools; GBGPU/BBHx local
  duplicates were deleted at Phase 3.dedup (2026-06-04); FEW's `std::`-qualified diverged copy is
  intentionally untouched.
- **`Backend.name` vs `Backend.backend_name`**: registry key vs. plugin-module name — easy to
  transpose; the `Backend` class docstring and the comment in `Backend.__reduce__` spell out
  this trap.
- **JAX backend (`gbt_jax`)** (added by commit `5622132`, "jax backend added"). It has no
  compiled plugin module (`_check_module_installed` is overridden to a no-op); `interpolate.py`
  special-cases it (`self.backend.name == "gbt_jax"`) because the JAX fit path is functional
  (returns coefficients) rather than in-place-mutating (like the C++ backends). It declares
  only `Feature.NUMPY` (`uses_gpu` is `False` even on a GPU) and its first load sets
  `jax_enable_x64` process-wide. No downstream package uses `gbt_jax`; LAT has its own
  `lisatools_jax`.
- **Bug notes live in docstrings.** The 2026-10 docstring pass documented present defects as
  `Warning:` / `Raises:` text (e.g. `Cuda11xBackend`, `globals.has_backend`, `Configuration`,
  `ConfigurationSetter`, `CubicSplineInterpolant.__call__` on JAX, the `exceptions` /
  `cutils` module notes); edit them when the defects are fixed.
- **`git status` on this repo is currently on branch `spline`**, not `main`, with an untracked
  `_version.py` (normal setuptools_scm artifact, not committed).

## 8. Where to look for X

| Want to... | Start in |
|---|---|
| Understand backend selection / registry mechanics | `src/gpubackendtools/gpubackendtools.py` (`Backend`, `BackendsManager`) |
| Understand the `Globals()` singleton / config / `get_backend` | `src/gpubackendtools/globals.py` |
| Add a `ParallelModuleBase` subclass in a downstream package | `src/gpubackendtools/parallelbase.py` (copy the `GBTParallelModuleBase` pattern) |
| Add a new native (C++/CUDA) backend method | `cutils/Interpolate.{cu,hh}` (impl) → `cutils/gbt_binding.cxx`/`.hpp` (nanobind) → `cutils/__init__.py` (`GBTBackendMethods` field + wire into each `GBT*Backend`) |
| Evaluate a spline from device code without linking GBT | `cutils/InterpolateDevice.hh` (`#include` it directly) |
| Fix/extend spline fit-and-solve (LAPACKE / PCR) | `cutils/Interpolate.cu` + `cutils/Interpolate.hh` |
| Debug nanobind binding / module naming | `cutils/gbt_binding.cxx` (`NB_MODULE(interp, m)`), `cutils/gbt_binding.hpp` (`array_type<T>`, `CubicSplineWrap`) |
| Trace CMake plugin-wheel layout (`gbt_backend_cuda12x` etc.) | `cutils/CMakeLists.txt` + `cutils/cmake_functions.cmake` (`apply_{cpu,gpu}_backend_common_options`) |
| Change LAPACKE detection/fetch behavior | `src/gpubackendtools/cutils/cmake_functions.cmake` (`get_lapacke`, `try_get_lapacke_with_{pkgconfig,cmake,cpm}`); options in the root `CMakeLists.txt` |
| Add/inspect the pure-JAX backend | `src/gpubackendtools/jax/backend.py` (`GBTJaxBackend`), `jax/cubic_spline.py` |
| Understand pickling/deepcopy of `Backend` | `src/gpubackendtools/gpubackendtools.py` (`Backend.__reduce__`, `__deepcopy__`, `__copy__`, `_resolve_backend_for_unpickle`) |
| Write a downstream backend (contract + example) | `docs/source/user/gpubackendtools.rst` ("Writing a downstream backend"); `src/gpubackendtools/cutils/__init__.py` is the in-tree template |
| Understand `CubicSplineInterpolant` usage/shape conventions | `src/gpubackendtools/interpolate.py`; `tests/test_cubic_spline.py` for worked examples |
| Understand the GPU macros shared across LISA Analysis Tools (`CUDA_SHARED`, `THREAD_START_X`, ...) | `cutils/gbt_global.h` |
| See how a downstream repo consumes GBT headers via CMake | `LISAanalysistools/src/lisatools/cutils/CMakeLists.txt`, the `GBT_CUTILS` `execute_process` block near the top (then `include(${GBT_CMAKE_TOOLS})` and the `gbt_global.h` check) |
| Configuration / env vars (`GPUBACKENDTOOLS_*` prefix) | `src/gpubackendtools/utils/config.py` (module docstring lists the entries); `docs/source/user/globals.rst` |
