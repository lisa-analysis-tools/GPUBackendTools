r"""JAX-native cubic-spline fit + evaluation.

This is the JAX analog of the C++ ``interpolate_wrap`` /
``CubicSplineWrap`` exposed by the ``gbt_backend_cpu`` /
``gbt_backend_cudaXXx`` plugin modules. Used by
:class:`gpubackendtools.interpolate.CubicSplineInterpolant` when the
selected backend is ``gbt_jax``.

**Fit**: ``scipy.interpolate.CubicSpline`` with ``bc_type='not-a-knot'``
(matches the C++ side). The fit runs once on the host at construction
time (a Python loop over the splines) and the resulting coefficients are
stashed as ``jnp`` arrays. Autograd flows through the evaluation, not the
fit -- which matches the use case (orbits / templates fit once, evaluated
many times with gradients). scipy is required.

**Eval**: pure JAX (:func:`evaluate_cubic_spline`, one spline per call; batch
it with ``jax.vmap`` if needed). ``jax.jit`` + ``jax.grad`` work, with
``derivative`` kept a static Python int.

The per-segment cubic form

.. math::

    y(x) = y_i + c_{1,i} (x - x_i) + c_{2,i} (x - x_i)^2 + c_{3,i} (x - x_i)^3

is the same one the C++ wrapper exposes.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------
# Single-spline fit (scipy under the hood). Host-side, NOT traced.
# --------------------------------------------------------------------
def _fit_one_not_a_knot_scipy(x_np: np.ndarray, y_np: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one not-a-knot cubic spline using scipy.

    Args:
        x_np: Increasing knots, shape ``(N,)``.
        y_np: Values at the knots, shape ``(N,)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``(c1, c2, c3)`` arrays of length
        ``N``. The last entry of each is unused at evaluation time (set to ``0.0``) but
        kept for layout parity with the C++ ``(ninterps*length,)`` storage.
    """
    from scipy.interpolate import CubicSpline as _SciCubicSpline
    sp = _SciCubicSpline(x_np, y_np, bc_type="not-a-knot")
    # scipy stores polynomial in (x - x_i) basis as:
    #   sp.c[3, i] = y_i  (= constant term)
    #   sp.c[2, i] = c1_i (first derivative)
    #   sp.c[1, i] = c2_i (1/2 second derivative)
    #   sp.c[0, i] = c3_i (1/6 third derivative)
    # shapes: sp.c has shape (4, N-1)
    c1_seg = sp.c[2, :]
    c2_seg = sp.c[1, :]
    c3_seg = sp.c[0, :]
    # Pad the last entry with 0 so storage matches y length N.
    c1 = np.concatenate([c1_seg, [0.0]])
    c2 = np.concatenate([c2_seg, [0.0]])
    c3 = np.concatenate([c3_seg, [0.0]])
    return c1, c2, c3


# --------------------------------------------------------------------
# Batched fit over ``ninterps`` independent splines.
# --------------------------------------------------------------------
def fit_cubic_spline_not_a_knot(x_flat, y_flat, length: int, ninterps: int
                                ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fit ``ninterps`` independent splines stacked in flat arrays.

    Inputs are ``(ninterps * length,)``; outputs are the same. Done
    host-side via scipy then transferred to JAX device.

    Args:
        x_flat (array_like): Knots of all splines, spline-major, shape
            ``(ninterps * length,)``; any array convertible with ``np.asarray``.
        y_flat (array_like): Values at the knots, same layout.
        length: Number of knots per spline.
        ninterps: Number of splines.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: ``(c1, c2, c3)``, each of shape
        ``(ninterps * length,)``; the last entry of every spline is ``0.0``.
    """
    x_np = np.asarray(x_flat).reshape(ninterps, length)
    y_np = np.asarray(y_flat).reshape(ninterps, length)
    c1 = np.empty_like(y_np)
    c2 = np.empty_like(y_np)
    c3 = np.empty_like(y_np)
    for i in range(ninterps):
        c1[i], c2[i], c3[i] = _fit_one_not_a_knot_scipy(x_np[i], y_np[i])
    return (
        jnp.asarray(c1.reshape(-1)),
        jnp.asarray(c2.reshape(-1)),
        jnp.asarray(c3.reshape(-1)),
    )


def interpolate_wrap_jax(x_flat, y_flat, B_flat, c1_flat, c2_flat, c3_flat,
                         length, ninterps):
    """JAX analog of ``interpolate_wrap``.

    The C++ signature mutates ``c1_flat`` / ``c2_flat`` / ``c3_flat``
    in place. JAX arrays are immutable, so this function instead
    returns the fitted ``(c1, c2, c3)`` and the caller is expected to
    rebind them. :class:`gpubackendtools.interpolate.CubicSplineInterpolant`
    branches on the JAX backend to use the returned values rather
    than re-using the input buffers.

    Args:
        x_flat (array_like): Knots, shape ``(ninterps * length,)``.
        y_flat (array_like): Values at the knots, same layout.
        B_flat (array_like): Unused (kept for signature parity).
        c1_flat (array_like): Unused.
        c2_flat (array_like): Unused.
        c3_flat (array_like): Unused.
        length (int): Number of knots per spline.
        ninterps (int): Number of splines.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: ``(c1, c2, c3)`` from
        :func:`fit_cubic_spline_not_a_knot`.
    """
    return fit_cubic_spline_not_a_knot(x_flat, y_flat, length, ninterps)


# --------------------------------------------------------------------
# Spline evaluation (pure JAX -- autograd-compatible).
# --------------------------------------------------------------------
def evaluate_cubic_spline(x: jnp.ndarray, y: jnp.ndarray,
                          c1: jnp.ndarray, c2: jnp.ndarray, c3: jnp.ndarray,
                          x_new: jnp.ndarray, derivative: int = 0
                          ) -> jnp.ndarray:
    """Evaluate one cubic spline at ``x_new``.

    The segment index is ``searchsorted(x, x_new, side="right") - 1`` clipped to
    ``[0, N - 2]``, so out-of-range ``x_new`` is NOT clamped: it is extrapolated with the
    first or last segment's cubic. Callers that want NaN / zero on out-of-bounds should
    mask the output.

    Args:
        x: Increasing knots, shape ``(N,)``.
        y: Values at the knots, shape ``(N,)``.
        c1: Linear coefficients, shape ``(N,)`` (e.g. one spline's slice of
            :func:`fit_cubic_spline_not_a_knot` output).
        c2: Quadratic coefficients, shape ``(N,)``.
        c3: Cubic coefficients, shape ``(N,)``.
        x_new: Query points (any shape).
        derivative: Derivative order, 0 to 3; matches
            :meth:`gpubackendtools.interpolate.CubicSplineInterpolant.__call__`.
            Default ``0``.

    Returns:
        jnp.ndarray: Values (or derivatives) with the shape of ``x_new``.

    Raises:
        ValueError: If ``derivative`` is not 0-3.

    Example:
        .. code-block:: python

            import jax.numpy as jnp
            from gpubackendtools.jax.cubic_spline import (
                evaluate_cubic_spline, fit_cubic_spline_not_a_knot,
            )

            x = jnp.linspace(0.0, 1.0, 50)
            y = jnp.sin(x)
            c1, c2, c3 = fit_cubic_spline_not_a_knot(x, y, length=50, ninterps=1)
            vals = evaluate_cubic_spline(x, y, c1, c2, c3, jnp.array([0.3, 0.7]))
    """
    N = x.shape[0]
    flat = x_new.reshape(-1)
    seg = jnp.clip(jnp.searchsorted(x, flat, side="right") - 1, 0, N - 2)
    x0 = x[seg]
    y0 = y[seg]
    c1s = c1[seg]
    c2s = c2[seg]
    c3s = c3[seg]
    dx = flat - x0
    if derivative == 0:
        out = y0 + c1s * dx + c2s * dx**2 + c3s * dx**3
    elif derivative == 1:
        out = c1s + 2.0 * c2s * dx + 3.0 * c3s * dx**2
    elif derivative == 2:
        out = 2.0 * c2s + 6.0 * c3s * dx
    elif derivative == 3:
        out = 6.0 * c3s
    else:
        raise ValueError("derivative must be 0..3")
    return out.reshape(x_new.shape)


# --------------------------------------------------------------------
# CubicSplineWrapJAX -- mirrors the C++ ``CubicSplineWrap`` API.
# --------------------------------------------------------------------
class CubicSplineWrapJAX:
    """Pure-Python analog of ``CubicSplineWrap`` / ``CubicSpline``.

    Constructor signature matches the C++ wrapper exactly so the
    existing call site

    .. code-block:: python

        backend.CubicSplineWrap(x_flat, y_flat, c1_flat, c2_flat,
                                c3_flat, ninterps, length, spline_type)

    works unchanged on the JAX backend. Only the constructor is mirrored: there is no
    ``eval_wrap``; evaluate with :meth:`evaluate_one`.

    Args:
        x_flat (array_like): Knots, shape ``(ninterps * length,)``.
        y_flat (array_like): Values at the knots, same layout.
        c1_flat (array_like): Linear coefficients, same layout.
        c2_flat (array_like): Quadratic coefficients, same layout.
        c3_flat (array_like): Cubic coefficients, same layout.
        ninterps: Number of splines.
        length: Number of knots per spline.
        spline_type: Spacing code (stored, not used).

    Attributes:
        x_flat (jnp.ndarray): Knots as a JAX array.
        y_flat (jnp.ndarray): Values as a JAX array.
        c1_flat (jnp.ndarray): Linear coefficients as a JAX array.
        c2_flat (jnp.ndarray): Quadratic coefficients as a JAX array.
        c3_flat (jnp.ndarray): Cubic coefficients as a JAX array.
        ninterps (int): Number of splines.
        length (int): Number of knots per spline.
        spline_type (int): Spacing code.
    """

    def __init__(self, x_flat, y_flat, c1_flat, c2_flat, c3_flat,
                 ninterps: int, length: int, spline_type: int):
        self.x_flat = jnp.asarray(x_flat)
        self.y_flat = jnp.asarray(y_flat)
        self.c1_flat = jnp.asarray(c1_flat)
        self.c2_flat = jnp.asarray(c2_flat)
        self.c3_flat = jnp.asarray(c3_flat)
        self.ninterps = int(ninterps)
        self.length = int(length)
        self.spline_type = int(spline_type)

    def evaluate_one(self, interp_index: int, x_new, derivative: int = 0):
        """Evaluate spline ``interp_index`` at ``x_new``.

        Convenience for downstream consumers that don't want to manage
        the ``(ninterps*length,)`` layout by hand.

        Args:
            interp_index: Spline index in ``[0, ninterps)``.
            x_new (jnp.ndarray): Query points (any shape).
            derivative: Derivative order, 0 to 3. Default ``0``.

        Returns:
            jnp.ndarray: :func:`evaluate_cubic_spline` output, shape of ``x_new``.
        """
        start = int(interp_index) * self.length
        stop = start + self.length
        return evaluate_cubic_spline(
            self.x_flat[start:stop],
            self.y_flat[start:stop],
            self.c1_flat[start:stop],
            self.c2_flat[start:stop],
            self.c3_flat[start:stop],
            x_new, derivative=derivative,
        )
