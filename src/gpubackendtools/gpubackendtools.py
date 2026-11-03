"""Backend abstractions and the backend registry of ``gpubackendtools``.

A *backend* is one compute flavor of one package: an array module ``xp`` (``numpy``,
``cupy`` or ``jax.numpy``) plus the compiled (or pure-Python) callables of that package for
that device. This module holds the generic machinery every package in the stack builds on:

- :class:`Backend`: the abstract base, with :class:`Backend.Feature` flags, the registry
  name contract (``_name`` / ``_backend_name``) and the singleton copy / pickle semantics.
- :class:`CpuBackend`, :class:`Cuda11xBackend`, :class:`Cuda12xBackend`,
  :class:`Cuda13xBackend`: abstract per-device bases that run the availability checks
  (plugin module installed, CuPy usable, CUDA driver version, NVIDIA shared libraries).
- :class:`BackendMethods`: the dataclass base of the per-package method tables.
- :class:`BackendsManager`: the lazy registry (name -> :class:`BackendStatus`) held by
  :class:`~gpubackendtools.globals.Globals`.

A downstream package defines one concrete class per flavor, for example::

    class MyPkgCpuBackend(CpuBackend, MyPkgBackend):
        _name = "mypkg_cpu"                  # registry key, "<pkg>_<flavor>"
        _backend_name = "mypkg_backend_cpu"  # importable plugin module

        @staticmethod
        def cpu_methods_loader() -> MyPkgBackendMethods:
            import mypkg_backend_cpu.ext     # raise BackendUnavailableException on failure
            ...

and registers all flavors at import time with
``Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend, ...})``.
The concrete GBT classes are in :mod:`gpubackendtools.cutils` and
:mod:`gpubackendtools.jax.backend`; user-facing lookup functions are in
:mod:`gpubackendtools.globals`.
"""

from __future__ import annotations
import dataclasses
import enum
import pickle
import types
import typing
import abc
from .exceptions import GPUBACKENDTOOLSException


class BackendUnavailableException(GPUBACKENDTOOLSException):
    """Exception raised when the backend is not available.

    This is the class :meth:`BackendsManager._try_loading_backend` catches to mark a backend
    :class:`BackendStatusUnavailable`; backend constructors and loaders signal "cannot load
    here" by raising it (or a subclass). It is a different class from
    :class:`gpubackendtools.exceptions.BackendUnavailableException`, which the manager does
    NOT catch.
    """


class BackendNotInstalled(BackendUnavailableException):
    """Exception raised when the backend plugin module cannot be imported.

    Raised by :meth:`Backend._check_module_installed`.
    """


class MissingDependencies(BackendUnavailableException):
    """Exception raised when the backend has missing dependencies.

    ``str(exc)`` appends ready-to-run ``pip install`` / ``conda install`` hints built from
    the two lists.

    Args:
        *args: Forwarded to :class:`Exception` (the message).
        pip_deps: Packages to install in a pip-managed environment. Required,
            keyword-only; may be empty.
        conda_deps: Packages to install in a conda-managed environment. Required,
            keyword-only; may be empty.
        **kwargs: Forwarded to :class:`Exception`.
    """

    pip_deps: typing.List[str]
    """List of missing dependencies to install in pip-managed environments"""

    conda_deps: typing.List[str]
    """List of missing dependencies to install in conda-managed environments."""

    def __init__(
        self, *args, pip_deps: typing.List[str], conda_deps: typing.List[str], **kwargs
    ):
        self.pip_deps = pip_deps
        self.conda_deps = conda_deps
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        message = super().__str__()
        if self.pip_deps:
            message += """
    If you are using gpubackendtools in an environment managed using pip, run:
        $ pip install {}

""".format(", ".join(self.pip_deps))
        if self.conda_deps:
            message += """
    If you are using gpubackendtools in an environment managed using conda, run:
        $ conda install {}

""".format(", ".join(self.conda_deps))
        return message


class MissingDriver(BackendUnavailableException):
    """Exception raised when backend needs driver-like software to be installed.

    Raised by the CUDA checks when the NVIDIA driver is not loaded or its CUDA version is
    outside the backend's supported range.
    """


class SoftwareException(BackendUnavailableException):
    """Exception raised due to unexpected software error when loading the backend"""


class MissingHardware(BackendUnavailableException):
    """Exception raised when backend needs unavailable hardware."""


@dataclasses.dataclass
class BackendMethods:
    """Base dataclass of a backend's method table.

    Each package subclasses it and adds one field per native symbol it exposes (see
    :class:`gpubackendtools.cutils.GBTBackendMethods`); the per-flavor loaders
    (``cpu_methods_loader``, ``cuda12x_module_loader``, ...) build and return an instance.
    :class:`Backend` copies ``xp`` from it; the package mixin copies the other fields.

    Attributes:
        xp (types.ModuleType): Array module of the backend (``numpy``, ``cupy`` or
            ``jax.numpy``).
    """

    xp: types.ModuleType


def _resolve_backend_for_unpickle(backend_name: str) -> "Backend":
    """Registry lookup executed by ``pickle.loads()`` for a pickled Backend.

    :meth:`Backend.__reduce__` stores ``(this function, (<registry name>,))``
    in the pickle stream instead of the Backend's state, so unpickling
    re-resolves the name through the global BackendsManager rather than
    reconstructing an instance (a Backend is a process-wide singleton, never
    a value — see the ``__reduce__`` comment). Must stay module-level: pickle
    records it by qualified name.

    Backends register when their owning package is imported (``gbt_*`` at
    ``gpubackendtools`` import, ``lisatools_*`` at ``lisatools`` import, ...).
    When unpickling in a fresh process the owning package may not have been
    imported yet; the registry-name prefix doubles as the import name for
    every downstream package, so try importing it before giving up.

    Args:
        backend_name: Registry key of the backend (``Backend.name``, e.g. ``"gbt_cpu"``).

    Returns:
        Backend: The instance currently registered under ``backend_name``.

    Raises:
        ValueError: If the name is unknown and importing its prefix does not register it.
        ImportError: If the prefix package cannot be imported (raised with the original
            ``ValueError`` as context).
        BackendAccessException: If the backend is registered but disabled or unavailable
            in this process.
    """
    from .globals import get_backend

    try:
        return get_backend(backend_name)
    except ValueError:
        import importlib

        try:
            importlib.import_module(backend_name.split("_")[0])
        except ImportError:
            raise  # surface the original unknown-backend error context
        return get_backend(backend_name)


class Backend:
    """Abstract definition of a backend.

    A backend instance is a process-wide singleton owned by the
    :class:`BackendsManager` registry: it is built lazily on first lookup and then shared
    by every object that resolves the same registry name (a later
    :meth:`BackendsManager.add_backends` call resets the registry, after which the next
    lookup builds a new instance). Concrete subclasses must set two class attributes:

    - ``_name``: the registry key, ``"<pkg>_<flavor>"`` (``"gbt_cpu"``,
      ``"lisatools_cuda12x"``, ...). Exposed as :attr:`name`; this is what
      :func:`~gpubackendtools.globals.get_backend` and pickling use.
    - ``_backend_name``: the importable plugin module, ``"<pkg>_backend_<flavor>"``
      (``"gbt_backend_cpu"``). Exposed as :attr:`backend_name`; this is what
      :meth:`_check_module_installed` imports. Do not confuse it with :attr:`name`.

    Copy and pickle never duplicate a backend: ``copy.copy`` and ``copy.deepcopy`` return
    the same instance, and pickling stores only :attr:`name`, which unpickling resolves
    through the registry of the receiving process (importing the owning package if
    needed). Objects holding a backend (or its name) are therefore deepcopy- and
    pickle-safe; do not cache ``xp`` as a plain attribute elsewhere, expose it as a
    property derived from the backend.

    Args:
        name: Ignored (kept for the subclass call signature); the registry name comes from
            the ``_name`` class attribute.
        methods: Method table of the backend; only its ``xp`` field is read here.
        features: Feature flags of the backend.

    Attributes:
        xp (types.ModuleType): Array module of the backend.
        features (Backend.Feature): Feature flags.
    """

    xp: types.ModuleType
    """Reference to package handling the backend ndarrays (numpy or cupy for now)"""

    class Feature(enum.Flag):
        """Capability flags a backend declares (combine with ``|``).

        CPU backends declare ``NUMPY``; CUDA backends declare ``CUPY | CUDA | GPU``; the
        GBT JAX backend declares ``NUMPY``.
        """

        NUMPY = enum.auto()
        """Flag indicating that backend uses numpy.ndarray"""

        CUPY = enum.auto()
        """Flag indicating that backend uses cupy.ndarray"""

        CUDA = enum.auto()
        """Flag indicating that backend uses CUDA devices and offers a set_cuda_device() method."""

        GPU = enum.auto()
        """Flag indicating that backend uses GPU hardware"""

        NONE = 0
        """Special flag representing no activated feature"""

    features: Feature
    """List of Backend features used by a backend"""

    def __init__(self, name: str, methods: BackendMethods, features: Feature):
        # TODO: is this okay?
        # self.name = name
        self.xp = methods.xp
        self.features = features

    def __deepcopy__(self, memo):
        # Backends carry a module reference (``self.xp`` is the numpy /
        # cupy module) plus C-extension function objects, both of which
        # can't be pickled and therefore can't be deepcopied through the
        # default __reduce__ protocol. They're also conceptually
        # singletons: one Backend instance per (package, flavor) lives
        # in the global BackendsManager registry and is shared across
        # the whole process, so duplicating one would silently produce
        # a second "view" of the same kernels with no semantic value.
        # Return self so deepcopy of any settings tree that transitively
        # holds a Backend (DataResidualArray.force_backend, DomainSettings
        # .force_backend, ParallelModuleBase subclasses, ...) completes
        # without TypeError.
        return self

    def __copy__(self):
        # Same rationale as __deepcopy__: Backend is a process-wide
        # singleton, never a value to clone.
        return self

    def __reduce__(self):
        """Pickle as a registry-name reference, resolved by ``_resolve_backend_for_unpickle``."""
        # Same singleton rationale as __deepcopy__/__copy__, but for PICKLE,
        # which ignores those two hooks entirely: by default pickle walks
        # __dict__ and dies on the module reference in self.xp with
        # "TypeError: cannot pickle 'module' object" — the failure any object
        # graph that transitively holds a Backend used to hit (settings trees
        # sent over MPI, multiprocessing workers, cached states, ...).
        #
        # __reduce__ replaces that default with a reconstruction recipe
        # `(callable, args)`: the pickle stream stores ONLY the registry name,
        # and pickle.loads() runs
        # `_resolve_backend_for_unpickle(<registry name>)` to look the Backend
        # up in the global BackendsManager. A Backend is never serialized as a
        # value, only as a name to re-resolve:
        #   * same process: loads(dumps(b)) is b — the identical singleton
        #     (unless a later add_backends() call rebuilt the registry);
        #   * fresh process: loads() reconnects to THAT process's registered
        #     backend (importing the owning package first if needed).
        #
        # NB: the registry key is self.name (the same contract
        # ParallelModuleBase relies on: get_backend(backend.name) returns
        # this instance) — NOT self.backend_name, which holds the
        # plugin-MODULE name ("lisatools_backend_cpu" vs registry key
        # "lisatools_cpu").
        try:
            name = self.name
        except ValueError:
            raise pickle.PicklingError(
                f"Backend {type(self).__name__} has no registry name (_name "
                "unset); only registry-managed backends can be pickled."
            )
        return (_resolve_backend_for_unpickle, (name,))

    @staticmethod
    def _check_module_installed(backend_name: str, module_name: str):
        """Check that the module containing the backend implementation is installed.

        Args:
            backend_name: Name used in the error message.
            module_name: Plugin module to import (``Backend.backend_name``).

        Raises:
            BackendNotInstalled: If importing ``module_name`` raises ``ModuleNotFoundError``.
        """
        import importlib
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise BackendNotInstalled(
                "The '{}' backend is not installed.".format(backend_name)
            ) from e

    def supports(self, feature: Feature) -> bool:
        """Check whether a backend supports a given feature.

        Args:
            feature: Flag (or combination of flags) to test.

        Returns:
            bool: ``True`` if every flag in ``feature`` is set in :attr:`features`.
        """
        return feature in self.features

    @property
    def uses_gpu(self) -> bool:
        """Shortcut to check if a backend supports GPU"""
        return self.supports(feature=Backend.Feature.GPU)

    @property
    def uses_numpy(self) -> bool:
        """Shortcut to check if a backend makes use of NumPy"""
        return self.supports(feature=Backend.Feature.NUMPY)

    @property
    def uses_cupy(self) -> bool:
        """Shortcut to check if a backend makes use of CuPy"""
        return self.supports(feature=Backend.Feature.CUPY)

    @property
    def uses_cuda(self) -> bool:
        """Shortcut to check whether a backend uses CUDA devices"""
        return self.supports(feature=Backend.Feature.CUDA)

    @property
    def name(self) -> str:
        """Registry key of the backend (the ``_name`` class attribute, e.g. ``"gbt_cpu"``).

        Raises:
            ValueError: If the subclass does not define ``_name``.
        """
        if not hasattr(self, "_name"):
            raise ValueError("Child class must have _name attribute set.")
        return self._name

    @property
    def backend_name(self) -> str:
        """Plugin module name (the ``_backend_name`` class attribute, e.g. ``"gbt_backend_cpu"``).

        Raises:
            ValueError: If the subclass does not define ``_backend_name``.
        """
        if not hasattr(self, "_backend_name"):
            raise ValueError("Child class must have _backend_name attribute set.")
        return self._backend_name
    
    @property
    def backend_base(self) -> str:
        """Device flavor of the backend: the part of :attr:`name` after its last ``_``.

        For example ``"cuda12x"`` for ``"lisatools_cuda12x"`` and ``"cpu"`` for
        ``"gbt_cpu"``; the whole name if it has no ``_``. Passing it as ``force_backend`` to
        a package class that prefixes bare names (e.g.
        :class:`~gpubackendtools.parallelbase.GBTParallelModuleBase`) selects the same device
        in that package.
        """
        if "_" in self.name:
            return self.name.split("_")[-1]
        else:
            return self.name


# TODO check: keep/remove/document -- no callers; references undefined GPUBACKENDTOOLSBackendMethods
class GPUBACKENDTOOLSBackend:
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]

    def __init__(self, gpubackendtools_backend_methods):
        # set direct gpubackendtools methods
        # pass rest to general backend
        assert isinstance(
            gpubackendtools_backend_methods, GPUBACKENDTOOLSBackendMethods
        )

        self.get_ll = gpubackendtools_backend_methods.get_ll
        self.fill_global = gpubackendtools_backend_methods.fill_global


class CpuBackend(Backend, abc.ABC):
    """Abstract base of every CPU (NumPy) backend.

    Construction checks that the plugin module ``backend_name`` is importable, calls
    :meth:`cpu_methods_loader` and declares ``Backend.Feature.NUMPY``. A concrete subclass
    sets ``_name`` / ``_backend_name`` and implements :meth:`cpu_methods_loader`; see
    :class:`gpubackendtools.cutils.GBTCpuBackend`.

    Raises:
        BackendNotInstalled: If the plugin module is not installed.
        BackendUnavailableException: Whatever :meth:`cpu_methods_loader` raises.
    """

    @staticmethod
    @abc.abstractmethod
    def cpu_methods_loader() -> BackendMethods:
        """Import the CPU plugin module and return the backend method table.

        Must be overridden. Implementations raise
        :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` (or a
        subclass) when the module cannot be loaded, so the manager records the backend as
        unavailable.

        Returns:
            BackendMethods: The package's method table with ``xp=numpy``.
        """
        raise not NotImplementedError

    @staticmethod
    def check_numpy() -> typing.ModuleType:
        """Import and return ``numpy``.

        Returns:
            types.ModuleType: The ``numpy`` module.

        Raises:
            MissingDependencies: If numpy cannot be imported.
        """
        try:
            import numpy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cpu' backend requires numpy", pip_deps=["numpy"], conda_deps=["numpy"]
            ) from e

        return numpy

    def __init__(self):
        """Initialize the CPU backend"""
        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")

        self._check_module_installed(self.name, self.backend_name)

        Backend.__init__(
            self,
            name=self.name,
            methods=self.cpu_methods_loader(),
            features=Backend.Feature.NUMPY,
        )


class _CudaBackend(Backend):
    """Shared CUDA availability checks for the ``Cuda*xBackend`` classes.

    Provides :meth:`check_cuda_backend` (the full load sequence), the NVIDIA shared-library
    preloader and :meth:`set_cuda_device`.
    """

    @staticmethod
    def _get_cuda_version() -> typing.Tuple[int, int]:
        """Get the CUDA version supported by the installed driver, via ``pynvml``.

        Returns:
            tuple[int, int]: ``(major, minor)`` CUDA version of the driver.

        Raises:
            MissingDriver: If the NVIDIA driver is not loaded.
            SoftwareException: If NVML fails or lacks the version query.
        """
        try:
            import pynvml

            pynvml.nvmlInit()
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        except pynvml.NVMLError_DriverNotLoaded as e:
            raise MissingDriver(
                "CUDA Driver is missing. Ensure you installed it properly."
            ) from e
        except pynvml.NVMLError as e:
            raise SoftwareException(
                "CUDA driver exception: could not detect a CUDA version."
            ) from e
        except AttributeError as e:
            raise SoftwareException(
                "The NVIDIA Management Library (libnvml) does not support the expected method to detect CUDA version."
            ) from e

        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        return cuda_major, cuda_minor

    @staticmethod
    def _check_cupy_works(cuda_major: int):
        """Check that CuPy imports, targets CUDA ``cuda_major`` and can run a kernel.

        Args:
            cuda_major: Expected CUDA major version of the CuPy runtime.

        Raises:
            MissingDependencies: If CuPy is missing, built for another CUDA major version, or
                cannot compile (missing CUDA runtime).
            SoftwareException: If CuPy cannot query the runtime version or execute.
        """
        try:
            import cupy
            import cupy_backends.cuda
        except ImportError as e:
            raise MissingDependencies(
                "CuPy is missing.",
                pip_deps=["cupy-cuda{}x".format(cuda_major)],
                conda_deps=[],
            ) from e

        try:
            cupy_cuda_ver = cupy.cuda.runtime.runtimeGetVersion() // 1000
        except RuntimeError as e:
            raise SoftwareException("CuPy could not detect runtime version") from e

        if cupy_cuda_ver != cuda_major:
            raise MissingDependencies(
                "CuPy is installed but supports CUDA {} instead of {}".format(
                    cupy_cuda_ver, cuda_major
                ),
                pip_deps=["cupy-cuda{}x".format(cuda_major)],
                conda_deps=[],
            )

        try:
            _ = cupy.arange(1)
            _ = cupy.cuda.runtime.getDevice()
        except cupy.cuda.compiler.CompileException as e:
            raise MissingDependencies(
                "CuPy fails to run due to missing CUDA Runtime.",
                pip_deps=["nvidia-cuda-runtime-cu{}".format(cuda_major)],
                conda_deps=[],
            ) from e
        except (cupy_backends.cuda.api.runtime.CUDARuntimeError, RuntimeError) as e:
            raise SoftwareException("CuPy could not execute properly.") from e

    @dataclasses.dataclass
    class NvidiaSoLib:
        """Description of a NVidia .so dynamic library"""

        soname: str
        """Dynamic library name (eg libname.so.1)"""

        module_name: str
        """Name of the nvidia module containing the library"""

        pip_pkg: typing.Optional[str] = None
        """Name of a pip-installable package providing that library"""

        conda_pkg: typing.Optional[str] = None
        """Name of a conda-installable package providing that library"""

    @staticmethod
    def _try_import_nvidia_solib(libs: typing.Sequence[NvidiaSoLib]) -> None:
        """Try to load a set of Nvidia dynamic libraries with ``ctypes``.

        Each library is tried first by soname (system search path), then from the
        ``nvidia/<module_name>/lib/`` directory of the ``nvidia-*`` pip wheels.

        Args:
            libs: Libraries to load.

        Raises:
            MissingDependencies: If any library fails both attempts; lists the matching
                pip / conda packages.
        """
        import ctypes
        import importlib
        import pathlib

        from .exceptions import ExceptionGroup

        try:
            nvidia_root = pathlib.Path(
                importlib.import_module("nvidia").__file__
            ).parent
        except ModuleNotFoundError:
            nvidia_root = None

        failed_idx: typing.List[int] = []
        exceptions = []
        for idx, lib in enumerate(libs):
            try:
                ctypes.cdll.LoadLibrary(lib.soname)
                continue
            except OSError as e:
                exceptions.append(e)

            try:
                if nvidia_root is not None:
                    ctypes.cdll.LoadLibrary(
                        nvidia_root / lib.module_name / "lib" / lib.soname
                    )
                    continue
            except OSError as e:
                exceptions.append(e)

            failed_idx.append(idx)

        if failed_idx:
            raise MissingDependencies(
                "Could not load following NVidia libraries: {}".format(
                    ", ".join([libs[idx].soname for idx in failed_idx])
                ),
                pip_deps=[
                    libs[idx].pip_pkg
                    for idx in failed_idx
                    if libs[idx].pip_pkg is not None
                ],
                conda_deps=[
                    libs[idx].conda_pkg
                    for idx in failed_idx
                    if libs[idx].conda_pkg is not None
                ],
            ) from ExceptionGroup(
                "Following exceptions were raised while trying to load NVidia libraries",
                exceptions,
            )

    @classmethod
    def check_cuda_backend(
        cls,
        name: str,
        backend_module_name: str,
        cuda_min: tuple[int, int],  # Inclusive minimum
        cuda_max: tuple[int, int],  # Exclusive maximum
        module_loader: typing.Callable[[], None],  # Method loading
        dynlib_loader: typing.Optional[typing.Callable[[], None]] = None,
    ) -> BackendMethods:
        """Perform all tests to ensure that a CUDA backend can be used, and load it.

        The sequence is:

        1. import ``backend_module_name`` (the plugin wheel);
        2. check CuPy targets CUDA ``cuda_min[0]`` and runs;
        3. call ``module_loader()`` and return its result if it succeeds;
        4. otherwise check the driver CUDA version lies in ``[cuda_min, cuda_max)``;
        5. retry ``module_loader()``;
        6. if that fails and ``dynlib_loader`` is given, preload the NVIDIA shared libraries
           with it and call ``module_loader()`` a last time.

        Only :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` raised by
        ``module_loader`` triggers the fallback steps; any other exception propagates.

        Args:
            name: Short flavor name used in error messages (``"cuda12x"``).
            backend_module_name: Plugin module to import (``Backend.backend_name``).
            cuda_min: Inclusive minimum ``(major, minor)`` driver CUDA version.
            cuda_max: Exclusive maximum ``(major, minor)`` driver CUDA version.
            module_loader: Zero-argument callable returning the method table.
            dynlib_loader: Optional zero-argument callable that preloads shared libraries.
                Default ``None``.

        Returns:
            BackendMethods: The table returned by ``module_loader``.

        Raises:
            MissingDependencies: Plugin module missing (the pip hint always names
                ``gpubackendtools-cuda<N>x``), CuPy unusable, or shared libraries missing.
            MissingDriver: Driver CUDA version outside ``[cuda_min, cuda_max)``.
            SoftwareException: CuPy or NVML failures.
            BackendUnavailableException: ``module_loader`` still failing at the last step.
        """

        # 1. Check backend module is installed
        try:
            cls._check_module_installed(name, backend_module_name)
        except BackendNotInstalled as e:
            raise MissingDependencies(
                "GPUBACKENDTOOLS CUDA plugin is missing.",
                pip_deps=["gpubackendtools-cuda{}x".format(cuda_min[0])],
                conda_deps=[],
            ) from e

        # 2. Check CuPy works with expected cuda major version
        cls._check_cupy_works(cuda_major=cuda_min[0])

        # 3. Try to load directly
        try:
            return module_loader()
        except BackendUnavailableException:
            pass

        # 4. Get and check the CUDA version
        cuda_version = cls._get_cuda_version()

        def fmt_version(version: tuple[int, int]) -> str:
            return "{}.{}".format(version[0], version[1])

        if cuda_version < cuda_min:
            raise MissingDriver(
                "Cuda version is below minimum supported version (expected >= {} and < {}, got {})".format(
                    fmt_version(cuda_min),
                    fmt_version(cuda_max),
                    fmt_version(cuda_version),
                )
            )
        if cuda_version >= cuda_max:
            raise MissingDriver(
                "Cuda version is above maximum supported version (expected >= {} and < {}, got {})".format(
                    fmt_version(cuda_min),
                    fmt_version(cuda_max),
                    fmt_version(cuda_version),
                )
            )

        # 5. Retry to load module
        try:
            return module_loader()
        except BackendUnavailableException as e:
            if dynlib_loader is None:
                raise e

        # 6. module_loader failed but dynlib_loader is defined, let's try that
        dynlib_loader()
        return module_loader()

    def set_cuda_device(self, dev: int):
        """Set the current CUDA device with ``cupy.cuda.runtime.setDevice``.

        The CUDA runtime keeps the current device per host thread, so this affects every
        CuPy allocation and kernel launch made afterwards from the calling thread, for all
        packages. Only CUDA backends define this method.

        Args:
            dev: CUDA device ordinal.
        """
        from cupy.cuda.runtime import setDevice

        setDevice(dev)


class Cuda11xBackend(_CudaBackend, abc.ABC):
    """Abstract base of every CUDA 11.x (CuPy) backend.

    Construction runs :meth:`_CudaBackend.check_cuda_backend` with the driver range
    ``[11.2, 12.0)`` and declares ``Feature.CUPY | Feature.CUDA | Feature.GPU``. A concrete
    subclass sets ``_name`` / ``_backend_name`` and implements :meth:`cuda11x_module_loader`.

    Warning:
        ``__init__`` passes ``Cuda11xBackend.cuda11x_module_loader`` (this abstract method)
        to the checks, not the subclass override as the 12.x / 13.x classes do. Once the
        plugin module and CuPy checks pass, loading raises ``NotImplementedError``, which
        the registry does not catch.
    """

    @staticmethod
    @abc.abstractmethod
    def cuda11x_module_loader():
        """Import the CUDA 11.x plugin module and return the backend method table.

        Must be overridden; raise
        :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` on failure.

        Returns:
            BackendMethods: The package's method table with ``xp=cupy``.
        """
        raise NotImplementedError

    @staticmethod
    def cuda11x_dynlib_loader():
        """Preload the CUDA 11 shared libraries (Linux only; a no-op elsewhere).

        Loads ``libcudart``, ``libcublas``, ``libnvJitLink``, ``libcusparse`` and ``libnvrtc``
        ``.so.11`` from the system search path or the ``nvidia-*-cu11`` pip wheels.

        Raises:
            MissingDependencies: If a library cannot be loaded.
        """
        import sys

        if sys.platform == "linux":
            cuda11x_solibs = [
                _CudaBackend.NvidiaSoLib(
                    soname="libcudart.so.11",
                    module_name="cuda_runtime",
                    pip_pkg="nvidia-cuda-runtime-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcublas.so.11",
                    module_name="cublas",
                    pip_pkg="nvidia-cublas-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvJitLink.so.11",
                    module_name="nvjitlink",
                    pip_pkg="nvidia-nvjitlink-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcusparse.so.11",
                    module_name="cusparse",
                    pip_pkg="nvidia-cusparse-cu11",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvrtc.so.11",
                    module_name="cuda_nvrtc",
                    pip_pkg="nvidia-cuda-nvrtc-cu11",
                    conda_pkg=None,
                ),
                # TODO: check this??
                # _CudaBackend.NvidiaSoLib(
                #     soname="libcufftw.so.11",
                #     module_name="cufft",
                #     pip_pkg="nvidia-cufft-cu11",
                #     conda_pkg=None,
                # ),
            ]
            _CudaBackend._try_import_nvidia_solib(cuda11x_solibs)

    def __init__(self):
        """Initialize the CUDA 11.x backend."""

        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")

        name = "cuda11x"
        methods = self.check_cuda_backend(
            name=name,
            backend_module_name=self.backend_name,
            cuda_min=(11, 2),
            cuda_max=(12, 0),
            module_loader=Cuda11xBackend.cuda11x_module_loader,
            dynlib_loader=Cuda11xBackend.cuda11x_dynlib_loader,
        )
        Feature = Backend.Feature

        super().__init__(
            name=name,
            methods=methods,
            features=Feature.CUPY | Feature.CUDA | Feature.GPU,
        )


class Cuda12xBackend(_CudaBackend, abc.ABC):
    """Abstract base of every CUDA 12.x (CuPy) backend.

    Construction runs :meth:`_CudaBackend.check_cuda_backend` with the driver range
    ``[12.0, 13.0)`` and the subclass's :meth:`cuda12x_module_loader` /
    :meth:`cuda12x_dynlib_loader`, and declares ``Feature.CUPY | Feature.CUDA | Feature.GPU``.
    A concrete subclass sets ``_name`` / ``_backend_name`` and implements
    :meth:`cuda12x_module_loader`; see :class:`gpubackendtools.cutils.GBTCuda12xBackend`.
    """

    @staticmethod
    @abc.abstractmethod
    def cuda12x_module_loader():
        """Import the CUDA 12.x plugin module and return the backend method table.

        Must be overridden; raise
        :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` on failure.

        Returns:
            BackendMethods: The package's method table with ``xp=cupy``.
        """
        raise NotImplementedError

    @staticmethod
    def cuda12x_dynlib_loader():
        """Preload the CUDA 12 shared libraries (Linux only; a no-op elsewhere).

        Loads ``libcudart``, ``libcublas``, ``libnvJitLink``, ``libcusparse`` and ``libnvrtc``
        ``.so.12`` from the system search path or the ``nvidia-*-cu12`` pip wheels.

        Raises:
            MissingDependencies: If a library cannot be loaded.
        """
        import sys

        if sys.platform == "linux":
            cuda12x_solibs = [
                _CudaBackend.NvidiaSoLib(
                    soname="libcudart.so.12",
                    module_name="cuda_runtime",
                    pip_pkg="nvidia-cuda-runtime-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcublas.so.12",
                    module_name="cublas",
                    pip_pkg="nvidia-cublas-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvJitLink.so.12",
                    module_name="nvjitlink",
                    pip_pkg="nvidia-nvjitlink-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcusparse.so.12",
                    module_name="cusparse",
                    pip_pkg="nvidia-cusparse-cu12",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvrtc.so.12",
                    module_name="cuda_nvrtc",
                    pip_pkg="nvidia-cuda-nvrtc-cu12",
                    conda_pkg=None,
                ),
                # TODO: check this
                # _CudaBackend.NvidiaSoLib(
                #     soname="libcufftw.so.12",
                #     module_name="cufft",
                #     pip_pkg="nvidia-cufft-cu12",
                #     conda_pkg=None,
                # ),
            ]
            _CudaBackend._try_import_nvidia_solib(cuda12x_solibs)

    def __init__(self):
        """Initialize the CUDA 12.x backend."""

        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")

        name = "cuda12x"
        methods = self.check_cuda_backend(
            name=name,
            backend_module_name=self.backend_name,
            cuda_min=(12, 0),
            cuda_max=(13, 0),
            module_loader=self.__class__.cuda12x_module_loader,
            dynlib_loader=self.__class__.cuda12x_dynlib_loader,
        )
        Feature = Backend.Feature

        super().__init__(
            name=name,
            methods=methods,
            features=Feature.CUPY | Feature.CUDA | Feature.GPU,
        )
        
        
class Cuda13xBackend(_CudaBackend, abc.ABC):
    """Abstract base of every CUDA 13.x (CuPy) backend.

    Construction runs :meth:`_CudaBackend.check_cuda_backend` with the driver range
    ``[13.0, 14.0)`` and the subclass's :meth:`cuda13x_module_loader` /
    :meth:`cuda13x_dynlib_loader`, and declares ``Feature.CUPY | Feature.CUDA | Feature.GPU``.
    A concrete subclass sets ``_name`` / ``_backend_name`` and implements
    :meth:`cuda13x_module_loader`; see :class:`gpubackendtools.cutils.GBTCuda13xBackend`.
    """

    @staticmethod
    @abc.abstractmethod
    def cuda13x_module_loader():
        """Import the CUDA 13.x plugin module and return the backend method table.

        Must be overridden; raise
        :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` on failure.

        Returns:
            BackendMethods: The package's method table with ``xp=cupy``.
        """
        raise NotImplementedError

    @staticmethod
    def cuda13x_dynlib_loader():
        """Preload the CUDA 13 shared libraries (Linux only; a no-op elsewhere).

        Loads ``libcudart``, ``libcublas``, ``libnvJitLink``, ``libcusparse`` and ``libnvrtc``
        ``.so.13`` from the system search path or the ``nvidia-*-cu13`` pip wheels.

        Raises:
            MissingDependencies: If a library cannot be loaded.
        """
        import sys

        if sys.platform == "linux":
            cuda13x_solibs = [
                _CudaBackend.NvidiaSoLib(
                    soname="libcudart.so.13",
                    module_name="cuda_runtime",
                    pip_pkg="nvidia-cuda-runtime-cu13",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcublas.so.13",
                    module_name="cublas",
                    pip_pkg="nvidia-cublas-cu13",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvJitLink.so.13",
                    module_name="nvjitlink",
                    pip_pkg="nvidia-nvjitlink-cu13",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libcusparse.so.13",
                    module_name="cusparse",
                    pip_pkg="nvidia-cusparse-cu13",
                    conda_pkg=None,
                ),
                _CudaBackend.NvidiaSoLib(
                    soname="libnvrtc.so.13",
                    module_name="cuda_nvrtc",
                    pip_pkg="nvidia-cuda-nvrtc-cu13",
                    conda_pkg=None,
                ),
                # TODO: check this
                # _CudaBackend.NvidiaSoLib(
                #     soname="libcufftw.so.13",
                #     module_name="cufft",
                #     pip_pkg="nvidia-cufft-cu13",
                #     conda_pkg=None,
                # ),
            ]
            _CudaBackend._try_import_nvidia_solib(cuda13x_solibs)

    def __init__(self):
        """Initialize the CUDA 13.x backend."""

        if self.backend_name is None:
            raise ValueError("Child class must declare `backend_name` class attribute.")

        name = "cuda13x"
        methods = self.check_cuda_backend(
            name=name,
            backend_module_name=self.backend_name,
            cuda_min=(13, 0),
            cuda_max=(14, 0),
            module_loader=self.__class__.cuda13x_module_loader,
            dynlib_loader=self.__class__.cuda13x_dynlib_loader,
        )
        Feature = Backend.Feature

        super().__init__(
            name=name,
            methods=methods,
            features=Feature.CUPY | Feature.CUDA | Feature.GPU,
        )


class BackendStatus:
    """Base class for backend statuses in the backends manager"""


class BackendStatusUnloaded(BackendStatus):
    """Backend is not yet loaded"""


@dataclasses.dataclass
class BackendStatusLoaded(BackendStatus):
    """Backend loaded successfully.

    Attributes:
        instance (Backend): The constructed backend singleton.
    """

    instance: Backend


class BackendStatusDisabled(BackendStatus):
    """Backend has been disabled by configuration options"""


@dataclasses.dataclass
class BackendStatusUnavailable:
    """Backend could not be loaded due to exception.

    Attributes:
        reason (BackendUnavailableException): The exception raised by the constructor.
    """

    reason: BackendUnavailableException


class BackendAccessException(GPUBACKENDTOOLSException):
    """Exception raised by :class:`BackendsManager` if a requested backend cannot be accessed.

    Raised for disabled or unavailable backends; the load failure, when there is one, is
    attached as ``__cause__``.
    """


class BackendsManager:
    """Handles loading and accessing backend instances.

    One manager lives in :attr:`gpubackendtools.globals.Globals.backends_manager`; every
    package registers its backend classes there. The manager keeps two maps:

    - ``_known_backends`` (a CLASS attribute, shared by all managers): registry key ->
      backend class, filled by :meth:`add_backend` / :meth:`add_backends`;
    - ``_registry`` (per instance): registry key -> :class:`BackendStatus`. A backend
      starts :class:`BackendStatusUnloaded` and is constructed on its first lookup; the
      result (:class:`BackendStatusLoaded` or :class:`BackendStatusUnavailable`) is cached.

    Registration order is preserved and matters for :meth:`get_first_backend` with bare
    flavor names. Every :meth:`add_backends` call rebuilds ``_registry`` for all known
    backends, so backends loaded earlier are constructed again on their next lookup.

    Args:
        enabled_backends: Accepted but currently ignored; the registry starts empty.
            Default ``None``.

    Example:
        .. code-block:: python

            from gpubackendtools import Globals

            manager = Globals().backends_manager
            manager.add_backends(
                {"mypkg_cpu": MyPkgCpuBackend, "mypkg_cuda12x": MyPkgCuda12xBackend}
            )
            manager.has_backend("mypkg_cpu")  # True if the plugin loads
            manager.get_first_backend(["mypkg_cuda12x", "mypkg_cpu"]).name
    """

    _known_backends: typing.Dict = {}
    _registry: typing.Dict[str, BackendStatus]

    @property
    def backend_list(self) -> typing.List[str]:
        """Return the list of backend names.

        Returns:
            list[str]: Registry keys in registration order (empty until
            :meth:`add_backends` has run).
        """
        return [name for name in self._registry.keys()]

    def __init__(self, enabled_backends: typing.Optional[typing.Sequence[str]] = None):
        """Create an empty registry; ``enabled_backends`` is not used."""
        # TODO: how do we handle this as we go to other packages.
        self._registry = {}

    def _initialize_backend(
        self, enabled_backends: typing.Optional[typing.Sequence[str]] = None
    ):
        """Rebuild the registry with every known backend marked unloaded.

        Args:
            enabled_backends: If given, each listed backend is loaded immediately and
                every other one is disabled. Default ``None`` (nothing loaded or
                disabled); :meth:`add_backends` always passes ``None``.

        Raises:
            BackendAccessException: If an enabled backend cannot be loaded.
        """
        # TODO: how do we handle this as we go to other packages.

        self._registry = {
            name: BackendStatusUnloaded() for name in self._known_backends.keys()
        }

        if enabled_backends is not None:
            for backend_name in self.backend_list:
                if backend_name in enabled_backends:
                    status = self._try_loading_backend(backend_name)
                    if isinstance(status, BackendStatusUnavailable):
                        raise BackendAccessException(
                            "Backend '{}' is marked as enabled but cannot be loaded".format(
                                backend_name
                            )
                        ) from status.reason
                else:
                    self._disable_backend(backend_name)

    def _disable_backend(self, backend_name: str):
        """Mark a backend as disabled"""
        self._registry[backend_name] = BackendStatusDisabled()

    def _try_loading_backend(self, backend_name: str) -> BackendStatus:
        """
        Try to load a backend and return its new status.

        If current status is different from UNLOADED, return it directly. Otherwise the
        backend class is instantiated; a
        :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` is recorded as
        :class:`BackendStatusUnavailable`, any other exception propagates to the caller.

        Args:
            backend_name: Exact registry key.

        Returns:
            BackendStatus: The (possibly new) status of the backend.

        Raises:
            ValueError: If ``backend_name`` is not a registry key.
        """
        if backend_name not in self._registry:
            raise ValueError(
                "'{}' is not a valid backend name, expected any of: {}".format(
                    backend_name, ", ".join(self.backend_list)
                )
            )

        if not isinstance(
            current_status := self._registry[backend_name], BackendStatusUnloaded
        ):
            return current_status

        try:
            new_status = BackendStatusLoaded(
                instance=self._known_backends[backend_name]()
            )
        except BackendUnavailableException as e:
            new_status = BackendStatusUnavailable(reason=e)

        self._registry[backend_name] = new_status
        return new_status

    def get_backend(self, backend_name: str) -> Backend:
        """Get a backend instance or raise a BackendAccessException.

        No alias handling: ``backend_name`` must be an exact registry key. The ``"cuda"`` /
        ``"gpu"`` aliases are resolved by :func:`gpubackendtools.globals.get_backend`.

        Args:
            backend_name: Exact registry key (``"gbt_cpu"``).

        Returns:
            Backend: The loaded backend singleton.

        Raises:
            ValueError: If ``backend_name`` is not a registry key.
            BackendAccessException: If the backend is disabled or failed to load.
        """
        status = self._try_loading_backend(backend_name=backend_name)
        if isinstance(status, BackendStatusDisabled):
            raise BackendAccessException(
                "Backend '{}' cannot be accessed, it has been disabled.".format(
                    backend_name
                )
            )
        if isinstance(status, BackendStatusUnavailable):
            raise BackendAccessException(
                "Backend '{}' is unavailable. See previous error messages.".format(
                    backend_name
                )
            ) from status.reason

        assert isinstance(status, BackendStatusLoaded)
        return status.instance

    def has_backend(self, backend_name: str) -> bool:
        """Check if a backend is available.

        Tries to load it on first call.

        Args:
            backend_name: Exact registry key.

        Returns:
            bool: ``True`` if the backend is loaded, ``False`` if disabled or unavailable.

        Raises:
            ValueError: If ``backend_name`` is not a registry key.
        """
        return isinstance(
            self._try_loading_backend(backend_name=backend_name), BackendStatusLoaded
        )

    # TODO: need to figure this out!!!
    def get_first_backend(self, backends: typing.Sequence[str]) -> Backend:
        """Get first available backend from a list or raise BackendAccessException if none is.

        Each entry is either an exact registry key (``"lisatools_cuda12x"``) or a bare
        flavor (``"cuda12x"``). A bare flavor resolves to the FIRST registry key, in
        registration order, whose part after the last ``_`` equals it; because
        ``gpubackendtools`` registers ``gbt_*`` at import, a bare ``"cpu"`` therefore
        resolves to ``"gbt_cpu"``, not to the calling package's backend. Packages that
        want their own backends pass prefixed names (see
        :meth:`gpubackendtools.parallelbase.ParallelModuleBase.supported_backends`).

        Args:
            backends: Candidate names in order of preference; must not be empty.

        Returns:
            Backend: The first candidate that loads.

        Raises:
            AssertionError: If ``backends`` is empty.
            ValueError: If a candidate reached before a loadable one is neither a registry
                key nor a registered flavor.
            BackendAccessException: If every candidate is disabled or unavailable (the
                reasons are attached as an ``ExceptionGroup`` cause).
        """
        assert len(backends) > 0

        reasons: typing.List[BackendUnavailableException] = []
        for backend_name in backends:
            # check for backend

            run_through = True
            registry_ind = 0
            keys = list(self._registry.keys())
            while run_through and registry_ind < len(keys):
                name = keys[registry_ind]
                if name.split("_")[-1] == backend_name:
                    backend_name = name
                    run_through = False
                registry_ind += 1

            for name in self._registry.keys():
                if name.split("_")[-1] == backend_name:
                    backend_name = name

            if isinstance(
                status := self._try_loading_backend(backend_name=backend_name),
                BackendStatusLoaded,
            ):
                return status.instance
            if isinstance(status, BackendStatusUnavailable):
                reasons.append(status.reason)
            if isinstance(status, BackendStatusDisabled):
                reasons.append(
                    BackendAccessException(
                        "Backend '{}' is disabled.".format(backend_name)
                    )
                )

        from .exceptions import ExceptionGroup

        raise BackendAccessException(
            "Could not access any of the following backends which are either disabled or unavailable: {}".format(
                ", ".join(backends)
            )
        ) from ExceptionGroup(
            "The backends were not available for following reasons.", reasons
        )

    def add_backend(self, name: str, backend: Backend):
        """Record a backend class under a registry key, without refreshing the registry.

        The backend is not visible to lookups until :meth:`add_backends` (which rebuilds
        the registry) runs; prefer :meth:`add_backends`.

        Args:
            name: Registry key; should equal the class's ``_name``.
            backend: Backend CLASS (not an instance); it is instantiated on first lookup.
        """
        self._known_backends[name] = backend

    def add_backends(self, backends: typing.Dict[str, Backend]):
        """Register backend classes and rebuild the registry.

        This is how a package publishes its backends, once at import time. Every known
        backend (from every package) is reset to unloaded, so previously loaded instances
        are rebuilt on their next lookup.

        Args:
            backends: Mapping registry key -> backend CLASS, e.g.
                ``{"gbt_cpu": GBTCpuBackend, "gbt_cuda12x": GBTCuda12xBackend}``.

        Example:
            .. code-block:: python

                from gpubackendtools import Globals

                Globals().backends_manager.add_backends(
                    {"mypkg_cpu": MyPkgCpuBackend, "mypkg_cuda12x": MyPkgCuda12xBackend}
                )
        """
        for key, value in backends.items():
            self.add_backend(key, value)
        # TODO: configuration added here?
        self._initialize_backend()


# TODO check: keep/remove/document -- no callers in GBT or any downstream package
@dataclasses.dataclass
class GpuBackendToolConsumer:
    name: str = None
    compiled_backends: typing.Dict[str, Backend] = None


__all__ = ["BackendsManager", "BackendAccessException"]
