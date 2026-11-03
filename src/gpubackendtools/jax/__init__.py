"""JAX backend for gpubackendtools.

Mirrors the CPU / CUDA backends in :mod:`gpubackendtools.cutils` but
uses ``jax.numpy`` and pure-Python implementations of the spline-fit /
spline-eval kernels (the fit runs on the host through scipy; the evaluation
is JAX code and differentiable). Code consumes this backend the same way it
consumes the cuda/cpu ones -- via the ``backend.CubicSplineWrap`` /
``backend.interpolate_wrap`` slots on the resolved
:class:`~gpubackendtools.cutils.GBTBackend` instance (registry key ``gbt_jax``).
Downstream JAX backends (e.g. ``lisatools.jax.backend.LISAToolsJaxBackend``)
follow the same pattern for their own symbols.

The subpackage import is gated on ``import jax`` so users without
JAX keep working with the C++/CUDA backends unmodified: without JAX,
``__all__`` is empty and :class:`GBTJaxBackend` is not defined.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from .backend import GBTJaxBackend
    from .cubic_spline import (
        CubicSplineWrapJAX,
        fit_cubic_spline_not_a_knot,
        evaluate_cubic_spline,
        interpolate_wrap_jax,
    )

    __all__ = [
        "GBTJaxBackend",
        "CubicSplineWrapJAX",
        "fit_cubic_spline_not_a_knot",
        "evaluate_cubic_spline",
        "interpolate_wrap_jax",
    ]
else:
    __all__ = []
