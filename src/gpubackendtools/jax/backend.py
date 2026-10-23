"""``GBTJaxBackend`` -- the JAX entry into ``gpubackendtools``' backend registry.

Slots into :class:`gpubackendtools.cutils.GBTBackendMethods` so that
:class:`gpubackendtools.interpolate.CubicSplineInterpolant` (and any
downstream class subclassing :class:`GBTParallelModuleBase`) can pick
the JAX path via ``force_backend='jax'``.

There is no compiled plugin module for this backend -- the
implementations live in :mod:`gpubackendtools.jax.cubic_spline` as
pure JAX functions. We subclass :class:`Backend` directly (not
:class:`CpuBackend`) so the ``_check_module_installed`` importlib
check is skipped; ``import jax`` is what we verify instead.

Loading the backend (its first lookup) turns on ``jax_enable_x64`` in the global JAX
config, for the whole process.
"""
from __future__ import annotations

from gpubackendtools.exceptions import MissingDependency
from gpubackendtools.gpubackendtools import Backend

from ..cutils import GBTBackend, GBTBackendMethods


def _check_jax():
    """Import JAX, enable ``jax_enable_x64`` globally, and return ``jax.numpy``.

    Raises:
        TypeError: If JAX cannot be imported (the intended ``MissingDependency`` is built
            with keyword arguments it does not accept).
    """
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as e:
        raise MissingDependency(
            "'jax' backend requires jax (with jaxlib)",
            pip_deps=["jax", "jaxlib"],
            conda_deps=["jax"],
        ) from e
    from jax import config
    config.update("jax_enable_x64", True)
    return jnp


def _jax_methods_loader() -> GBTBackendMethods:
    """Build the JAX method table from :mod:`gpubackendtools.jax.cubic_spline`.

    ``interpolate_wrap`` is :func:`~gpubackendtools.jax.cubic_spline.interpolate_wrap_jax`,
    ``fit_cubic_spline_thomas`` is
    :func:`~gpubackendtools.jax.cubic_spline.fit_cubic_spline_not_a_knot`, both
    ``CubicSplineWrap`` and ``CubicSpline`` are
    :class:`~gpubackendtools.jax.cubic_spline.CubicSplineWrapJAX`, and
    ``fit_cubic_spline_pcr`` is ``None``.

    Returns:
        GBTBackendMethods: Table with ``xp=jax.numpy``.
    """
    jnp = _check_jax()
    from .cubic_spline import CubicSplineWrapJAX, fit_cubic_spline_not_a_knot, interpolate_wrap_jax

    class _CubicSplineBase:
        """Placeholder for the C++ ``CubicSpline`` base class.

        The C++ exposes both a wrapper class (``CubicSplineWrap``) and
        the underlying ``CubicSpline`` C++ class. Downstream code only
        touches the wrapper, but the backend dataclass declares the
        base too. We expose ``CubicSplineWrapJAX`` for both slots.
        """
        pass

    return GBTBackendMethods(
        interpolate_wrap=interpolate_wrap_jax,
        fit_cubic_spline_thomas=fit_cubic_spline_not_a_knot,
        fit_cubic_spline_pcr=None,
        CubicSplineWrap=CubicSplineWrapJAX,
        CubicSpline=CubicSplineWrapJAX,
        xp=jnp,
    )


class GBTJaxBackend(Backend, GBTBackend):
    """JAX backend object plugged into ``Globals().backends_manager`` (key ``gbt_jax``).

    Registered by ``import gpubackendtools`` when JAX is importable. Declares
    ``Backend.Feature.NUMPY`` only, so ``uses_gpu`` is ``False`` even when JAX runs on a
    GPU. Construction loads the method table (enabling ``jax_enable_x64``).

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Example:
        .. code-block:: python

            from gpubackendtools.interpolate import CubicSplineInterpolant

            spl = CubicSplineInterpolant(x, y, force_backend="jax")  # backend gbt_jax
    """

    _backend_name: str = "gbt_backend_jax"
    _name = "gbt_jax"

    def __init__(self, *args, **kwargs):
        methods = _jax_methods_loader()
        Backend.__init__(
            self,
            name=self._name,
            methods=methods,
            features=Backend.Feature.NUMPY,
        )
        GBTBackend.__init__(self, methods)

    @staticmethod
    def _check_module_installed(*args, **kwargs):
        """No-op: there is no plugin module to check."""
        return None
