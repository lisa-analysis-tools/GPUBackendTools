"""Concrete ``gpubackendtools`` backends (``gbt_cpu``, ``gbt_cuda11x/12x/13x``).

Each class loads the nanobind module ``interp`` of its plugin package
(``gbt_backend_cpu``, ``gbt_backend_cuda11x``, ``gbt_backend_cuda12x``,
``gbt_backend_cuda13x``) and exposes its symbols through :class:`GBTBackendMethods` /
:class:`GBTBackend`. They are registered by ``import gpubackendtools`` and are the template
downstream packages copy for their own backends (``lisatools.cutils``, ``gbgpu.cutils``,
...). To add a native function: bind it in ``gbt_binding.cxx``, add a field to
:class:`GBTBackendMethods`, copy it in :meth:`GBTBackend.__init__` and fill it in every
loader.

Note:
    The loaders raise :class:`gpubackendtools.exceptions.BackendUnavailableException`
    (star-imported from :mod:`gpubackendtools.exceptions`), which the registry does not
    catch, and the CUDA loaders reference ``MissingDependencies``, which that star import
    does not provide.
"""

from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union

from ..gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend, Cuda13xBackend
from ..exceptions import *

@dataclasses.dataclass
class GBTBackendMethods(BackendMethods):
    """Method table of a ``gpubackendtools`` backend.

    Attributes:
        xp (types.ModuleType): Array module (``numpy``, ``cupy`` or ``jax.numpy``).
        interpolate_wrap (Callable): Batched spline fit
            ``(x, y, B, upper_diag, diag, lower_diag, length, ninterps)``; see
            :attr:`gpubackendtools.interpolate.CubicSplineInterpolant.interpolate_arrays`.
        fit_cubic_spline_thomas (Callable or None): Single-spline Thomas-algorithm fit
            ``(x, y, c1, c2, c3, B, length, spline_type)``, in place; CPU only (``None`` on
            CUDA; the scipy fit ``fit_cubic_spline_not_a_knot`` on JAX, with another
            signature).
        fit_cubic_spline_pcr (Callable or None): Single-spline parallel-cyclic-reduction
            fit, same signature as the Thomas one; CUDA only (``None`` elsewhere).
        CubicSplineWrap (type): Native spline wrapper class (``CubicSplineWrapCPU`` /
            ``CubicSplineWrapGPU``), constructed from
            :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.cpp_class_args`.
        CubicSpline (type): Underlying native ``CubicSplineCPU`` / ``CubicSplineGPU`` class
            (registered for type checks; not constructible from Python).
    """

    interpolate_wrap: typing.Callable[(...), None]
    fit_cubic_spline_thomas: typing.Callable[(...), None]
    fit_cubic_spline_pcr: typing.Optional[typing.Callable[(...), None]]
    CubicSplineWrap: object
    CubicSpline: object

class GBTBackend:
    """Mixin copying the :class:`GBTBackendMethods` fields onto the backend instance.

    Concrete backends inherit from a device base (``CpuBackend``, ``Cuda12xBackend``, ...)
    and from this class, so ``backend.interpolate_wrap``, ``backend.CubicSplineWrap``, ...
    are plain attributes.

    Args:
        gbt_backend_methods (GBTBackendMethods): The loaded method table.

    Raises:
        AssertionError: If ``gbt_backend_methods`` is not a :class:`GBTBackendMethods`.
    """

    interpolate_wrap: typing.Callable[(...), None]
    fit_cubic_spline_thomas: typing.Callable[(...), None]
    fit_cubic_spline_pcr: typing.Optional[typing.Callable[(...), None]]
    CubicSplineWrap: object
    CubicSpline: object

    def __init__(self, gbt_backend_methods):

        # set direct gbt methods
        # pass rest to general backend
        assert isinstance(gbt_backend_methods, GBTBackendMethods)

        self.interpolate_wrap = gbt_backend_methods.interpolate_wrap
        self.fit_cubic_spline_thomas = gbt_backend_methods.fit_cubic_spline_thomas
        self.fit_cubic_spline_pcr = gbt_backend_methods.fit_cubic_spline_pcr
        self.CubicSplineWrap = gbt_backend_methods.CubicSplineWrap
        self.CubicSpline = gbt_backend_methods.CubicSpline


class GBTCpuBackend(CpuBackend, GBTBackend):
    """Implementation of the CPU backend (registry key ``gbt_cpu``).

    Loads ``gbt_backend_cpu.interp`` with ``xp=numpy``; the CPU fit uses LAPACKE.

    Example:
        >>> import gpubackendtools
        >>> from gpubackendtools import get_backend, has_backend
        >>> if has_backend("gbt_cpu"):
        ...     fit = get_backend("gbt_cpu").fit_cubic_spline_thomas
    """
    
    _backend_name = "gbt_backend_cpu"
    _name = "gbt_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> GBTBackendMethods:
        """Import ``gbt_backend_cpu.interp`` and build the CPU method table.

        Returns:
            GBTBackendMethods: Table with ``xp=numpy`` and ``fit_cubic_spline_pcr=None``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If the module cannot be
                imported.
            MissingDependencies: If numpy is missing.
        """
        try:
            import gbt_backend_cpu.interp
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = GBTCpuBackend.check_numpy()

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cpu.interp.interpolate_wrap,
            fit_cubic_spline_thomas=gbt_backend_cpu.interp.fit_cubic_spline_thomas,
            fit_cubic_spline_pcr=None,
            CubicSplineWrap=gbt_backend_cpu.interp.CubicSplineWrapCPU,
            CubicSpline=gbt_backend_cpu.interp.CubicSplineCPU,
            xp=numpy,
        )


class GBTCuda11xBackend(Cuda11xBackend, GBTBackend):

    """Implementation of CUDA 11.x backend (registry key ``gbt_cuda11x``).

    Loads ``gbt_backend_cuda11x.interp`` with ``xp=cupy``. See the warning on
    :class:`~gpubackendtools.gpubackendtools.Cuda11xBackend`: its constructor calls the
    abstract loader, so this backend cannot finish loading.
    """
    _backend_name : str = "gbt_backend_cuda11x"
    _name = "gbt_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        """Import ``gbt_backend_cuda11x.interp`` and build the CUDA 11.x method table.

        Returns:
            GBTBackendMethods: Table with ``xp=cupy`` and ``fit_cubic_spline_thomas=None``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If the module cannot be
                imported.
        """
        try:
            import gbt_backend_cuda11x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda11x.interp.interpolate_wrap,
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda11x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda11x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda11x.interp.CubicSplineGPU,
            xp=cupy,
        )

class GBTCuda12xBackend(Cuda12xBackend, GBTBackend):
    """Implementation of CUDA 12.x backend (registry key ``gbt_cuda12x``).

    Loads ``gbt_backend_cuda12x.interp`` with ``xp=cupy`` (driver CUDA ``[12.0, 13.0)``).
    """
    _backend_name : str = "gbt_backend_cuda12x"
    _name = "gbt_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        """Import ``gbt_backend_cuda12x.interp`` and build the CUDA 12.x method table.

        Returns:
            GBTBackendMethods: Table with ``xp=cupy`` and ``fit_cubic_spline_thomas=None``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If the module cannot be
                imported.
        """
        try:
            import gbt_backend_cuda12x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda12x.interp.interpolate_wrap,
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda12x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda12x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda12x.interp.CubicSplineGPU,
            xp=cupy,
        )


class GBTCuda13xBackend(Cuda13xBackend, GBTBackend):
    """Implementation of CUDA 13.x backend (registry key ``gbt_cuda13x``).

    Loads ``gbt_backend_cuda13x.interp`` with ``xp=cupy`` (driver CUDA ``[13.0, 14.0)``).
    """
    _backend_name : str = "gbt_backend_cuda13x"
    _name = "gbt_cuda13x"
    
    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        GBTBackend.__init__(self, self.cuda13x_module_loader())
        
    @staticmethod
    def cuda13x_module_loader():
        """Import ``gbt_backend_cuda13x.interp`` and build the CUDA 13.x method table.

        Returns:
            GBTBackendMethods: Table with ``xp=cupy`` and ``fit_cubic_spline_thomas=None``.

        Raises:
            gpubackendtools.exceptions.BackendUnavailableException: If the module cannot be
                imported.
        """
        try:
            import gbt_backend_cuda13x.interp

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda13x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda13x' backend requires cupy", pip_deps=["cupy-cuda13x"]
            ) from e

        return GBTBackendMethods(
            interpolate_wrap=gbt_backend_cuda13x.interp.interpolate_wrap,
            fit_cubic_spline_thomas=None,
            fit_cubic_spline_pcr=gbt_backend_cuda13x.interp.fit_cubic_spline_pcr,
            CubicSplineWrap=gbt_backend_cuda13x.interp.CubicSplineWrapGPU,
            CubicSpline=gbt_backend_cuda13x.interp.CubicSplineGPU,
            xp=cupy,
        )

"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


