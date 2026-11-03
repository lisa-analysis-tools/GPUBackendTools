"""Base classes for "compute objects" that pick a backend once, at construction.

:class:`ParallelModuleBase` is the class every backend-dispatching class of the stack derives
from (directly, or through a package base such as
``lisatools.utils.parallelbase.LISAToolsParallelModule``, ``gbgpu.parallelbase.GBGPUParallelModule``
or ``few.utils.baseclasses.FewBackendConsumer``). :class:`GBTParallelModuleBase` is the
``gpubackendtools`` flavor, which prefixes ``"gbt_"`` onto bare backend strings; its in-tree
user is :class:`~gpubackendtools.interpolate.CubicSplineInterpolant`.
"""

from .globals import get_backend, get_first_backend
from .gpubackendtools import Backend
from typing import Optional, Sequence, TypeVar
import types


class ParallelModuleBase:  # (Citable):
    """
    Base class for modules that can use a GPU (or revert back to CPU).

    This class mainly handles backend selection. Each backend offers accelerated
    computations on a specific device (CPU, CUDA 11.x / 12.x / 13.x enabled GPU, or JAX).
    The choice is made once, at construction, and stored as the backend's registry NAME
    (``self._backend_name``); :attr:`backend` and :attr:`xp` re-resolve it through the
    global registry on every access, so instances deepcopy and pickle cleanly. Never cache
    the array module as an attribute (``self.xp = cupy``); use the :attr:`xp` property.

    ``force_backend`` is resolved as follows:

    - ``None``: the first loadable entry of :meth:`supported_backends` (via
      :func:`~gpubackendtools.globals.get_first_backend`);
    - ``str``: passed to :func:`~gpubackendtools.globals.get_backend`, so it must be a full
      registry name (``"gbt_cpu"``) or a ``"<pkg>_cuda"`` / ``"<pkg>_gpu"`` alias. Bare
      flavors (``"cpu"``, ``"cuda12x"``, ``"jax"``, ``"cuda"``, ``"gpu"``) work only in
      subclasses that add their package prefix (:class:`GBTParallelModuleBase`,
      ``LISAToolsParallelModule``, ...);
    - :class:`~gpubackendtools.gpubackendtools.Backend` instance: its registry
      :attr:`~gpubackendtools.gpubackendtools.Backend.name` is used as is;
    - ``(package, flavor)`` tuple or list of two strings: joined as ``"package_flavor"``.

    Subclass contract: implement the classmethod :meth:`supported_backends`, returning
    candidate names in order of preference. Return PREFIXED names
    (``["mypkg_" + f for f in cls.GPU_RECOMMENDED()]``): a bare flavor list makes the
    default pick the first registered package with that flavor (``gbt_*``), not yours.

    Args:
        force_backend (str | Backend | tuple[str, str], optional): Backend to use, see
            above. Default ``None`` picks from :meth:`supported_backends`.

    Raises:
        ValueError: If ``force_backend`` has an unsupported type or names an unregistered
            backend.
        BackendAccessException: If the requested backend (or every supported one) is
            disabled or unavailable.

    Example:
        .. code-block:: python

            from gpubackendtools import ParallelModuleBase


            class DemoNormFFT(ParallelModuleBase):
                @classmethod
                def supported_backends(cls):
                    return ["gbt_" + flavor for flavor in cls.GPU_RECOMMENDED()]

                def __call__(self, x):
                    xp = self.xp
                    x = xp.asarray(x)
                    return xp.fft.rfft((x - xp.mean(x)) / xp.std(x))


            module = DemoNormFFT(force_backend="gbt_cpu")
            module.backend.name  # "gbt_cpu"
            child = module.build_with_same_backend(DemoNormFFT)
    """

    _backend_name: str

    def __init__(self, /, force_backend=None):
        if force_backend is None:
            self._backend_name = get_first_backend(self.supported_backends()).name

        elif isinstance(force_backend, str):
            self._backend_name = get_backend(force_backend).name
        elif isinstance(force_backend, Backend):
            force_backend = force_backend.name
            self._backend_name = get_backend(force_backend).name
        elif isinstance(force_backend, tuple) or isinstance(force_backend, list):
            # comes in the form of (module, backend)
            assert isinstance(force_backend[0], str) and isinstance(
                force_backend[1], str
            )
            force_backend = force_backend[0] + "_" + force_backend[1]
            self._backend_name = get_backend(force_backend).name

        else:
            raise ValueError("FIX THIS")

    @property
    def backend(self) -> Backend:
        """Access the underlying backend (looked up by registry name on each access)."""
        return get_backend(self._backend_name)

    @classmethod
    def supported_backends(cls) -> Sequence[str]:
        """List of backends supported by a parallel module by order of preference.

        Must be overridden. Build it from one of the flavor helpers below, prefixed with
        the package namespace, e.g. ``["gbt_" + f for f in cls.GPU_RECOMMENDED()]``.

        Returns:
            Sequence[str]: Candidate backend names, most preferred first.

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError(
            "Class {} does not implement the supported_backends method.".format(cls)
        )

    @staticmethod
    def CPU_ONLY() -> list[str]:
        """List of supported backend for CPU only class: ``["cpu"]`` (bare flavors)."""
        return ["cpu"]

    @staticmethod
    def GPU_RECOMMENDED() -> list[str]:
        """Bare flavors for a GPU-recommended class with CPU support.

        Returns:
            list[str]: ``["cuda13x", "cuda12x", "cuda11x", "cpu"]``.
        """
        return ["cuda13x", "cuda12x", "cuda11x", "cpu"]

    @staticmethod
    def CPU_RECOMMENDED_WITH_GPU_SUPPORT() -> list[str]:
        """Bare flavors for a CPU-recommended class with GPU support.

        Returns:
            list[str]: ``["cpu", "cuda13x", "cuda12x", "cuda11x"]``.
        """
        return ["cpu", "cuda13x", "cuda12x", "cuda11x"]

    @staticmethod
    def GPU_ONLY() -> list[str]:
        """Bare flavors for a GPU-only class.

        Returns:
            list[str]: ``["cuda13x", "cuda12x", "cuda11x"]``.
        """
        return ["cuda13x", "cuda12x", "cuda11x"]

    @property
    def xp(self) -> types.ModuleType:
        """Return the module providing ndarray capabilities (``self.backend.xp``)."""
        return self.backend.xp

    @property
    def backend_name(self) -> str:
        """Return the registry name of the current backend (e.g. ``"gbt_cpu"``)."""
        return self.backend.name

    ParallelModuleDerivate = TypeVar(
        "ParallelModuleDerivate", bound="ParallelModuleBase"
    )

    def build_with_same_backend(
        self,
        module_class: type[ParallelModuleDerivate],
        args: Optional[list] = None,
        kwargs: Optional[dict] = None,
    ) -> ParallelModuleDerivate:
        """
        Build an instance of ``module_class`` with same backend as current object.

        Args:
            module_class: Class of the object to be built; must derive from
                :class:`ParallelModuleBase` and accept a ``force_backend`` keyword.
            args: Positional arguments for the ``module_class`` constructor. Default
                ``None`` (none).
            kwargs: Keyword arguments for the ``module_class`` constructor; any
                ``force_backend`` entry is replaced by :attr:`backend_name`, and the dict is
                modified in place. Default ``None`` (none).

        Returns:
            ParallelModuleBase: The new ``module_class`` instance.

        Example:
            .. code-block:: python

                child = module.build_with_same_backend(OtherModule, args=[x], kwargs={"n": 4})
        """
        args = [] if args is None else args
        return module_class(*args, **self.adapt_backend_kwargs(kwargs=kwargs))

    def adapt_backend_kwargs(self, kwargs: Optional[dict] = None) -> dict:
        """Adapt a set of keyword arguments to add/set 'force_backend' to current backend.

        Args:
            kwargs: Keyword arguments; modified in place. Default ``None`` (a new dict).

        Returns:
            dict: ``kwargs`` with ``force_backend`` set to :attr:`backend_name` (the full
            registry name).
        """
        if kwargs is None:
            kwargs = {}
        kwargs["force_backend"] = self.backend_name
        return kwargs


class GBTParallelModuleBase(ParallelModuleBase):
    """:class:`ParallelModuleBase` scoped to the ``gpubackendtools`` (``gbt_*``) backends.

    A string ``force_backend`` that does not start with ``"gbt_"`` gets the prefix, so the
    bare flavors ``"cpu"``, ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"jax"`` and the
    aliases ``"cuda"`` / ``"gpu"`` (first available CUDA flavor) select GBT's backends.
    Other ``force_backend`` forms are passed through unchanged. Downstream packages copy
    this pattern with their own ``_BACKEND_PREFIX``.

    Args:
        force_backend (str | Backend | tuple[str, str], optional): See
            :class:`ParallelModuleBase`. Default ``None``.
    """

    _BACKEND_PREFIX = "gbt"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = self._BACKEND_PREFIX + "_" + force_backend
        super().__init__(force_backend)
