Cubic-Spline Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~gpubackendtools.interpolate.CubicSplineInterpolant` is the reference compute object
of ``gpubackendtools``: it fits many independent not-a-knot cubic splines (the same splines
as ``scipy.interpolate.CubicSpline`` with its default boundary condition) once, at
construction, and evaluates them with array code on the backend's ``xp``.

- **Input layout.** ``x`` and ``y_all`` have the same shape; the last axis holds the knots of
  one spline and the leading axes enumerate the splines. Flattened 1D input needs
  ``ninterps`` and ``length``.
- **Backends.** ``force_backend`` takes the bare flavors (``"cpu"``, ``"cuda11x"``,
  ``"cuda12x"``, ``"cuda13x"``, ``"jax"``) or the ``"cuda"`` / ``"gpu"`` aliases; ``"gbt_"``
  is prepended to a string that does not already start with it (see
  :class:`~gpubackendtools.parallelbase.GBTParallelModuleBase`). The
  default is the first available of ``gbt_cuda13x``, ``gbt_cuda12x``, ``gbt_cuda11x``,
  ``gbt_cpu``, ``gbt_jax``. The native fit solves the tridiagonal systems with LAPACKE
  ``dgtsv`` on CPU and cuSPARSE ``gtsv2StridedBatch`` on CUDA; on ``gbt_jax`` the fit runs on
  the host through scipy.
- **Evaluation.** ``spline(x_new)`` takes one row of query points per spline;
  ``ind_interps`` selects a subset of splines (unique indices, one row of ``x_new`` each).
  ``derivative`` returns the first to third derivative. Points outside a spline's knot range
  raise ``ValueError`` by default; with ``error_out_of_bounds=False`` they produce a warning
  and the value ``0.0`` (they are not clamped to the end values).
- **Knot spacing.** The spacing of ``x`` is detected and stored as
  :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.spline_type`, with the module
  constants ``CUBIC_SPLINE_LINEAR_SPACING`` (1), ``CUBIC_SPLINE_LOG10_SPACING`` (2) and
  ``CUBIC_SPLINE_GENERAL_SPACING`` (3), the same codes as ``InterpolateDevice.hh``.
- **Native consumers.** :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.container`
  and :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.cpp_class_args` hand the flat
  coefficient arrays to compiled code;
  :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.cpp_class` builds the backend's
  native ``CubicSplineWrap`` over them. C++ / CUDA code can evaluate the splines with the
  header-only evaluators of ``InterpolateDevice.hh`` (directory:
  :func:`gpubackendtools.get_include`).

.. automodule:: gpubackendtools.interpolate

.. autoclass:: gpubackendtools.interpolate.CubicSplineInterpolant
    :members:
    :show-inheritance:
    :inherited-members:
    :exclude-members: ParallelModuleDerivate

.. autofunction:: gpubackendtools.interpolate.searchsorted2d_vec

Native module ``gbt_backend_<flavor>.interp``
---------------------------------------------

The nanobind extension behind the CPU and CUDA backends (source ``cutils/gbt_binding.cxx``).
The backend exposes its symbols as attributes, for example
``get_backend("gbt_cpu").interpolate_wrap`` (see
:class:`~gpubackendtools.cutils.GBTBackendMethods`). Array arguments are float64 host arrays
on the CPU build and CuPy device arrays on the CUDA builds; a size mismatch raises
``ValueError``.

- ``interpolate_wrap(x, y, B, upper_diag, diag, lower_diag, length, ninterps)``: batched fit
  on flat arrays of size ``length * ninterps``; fills ``upper_diag``, ``diag`` and
  ``lower_diag`` with ``c1``, ``c2`` and ``c3`` in place.
- ``fit_cubic_spline_thomas(x, y, c1, c2, c3, B, length, spline_type)`` (CPU build only) and
  ``fit_cubic_spline_pcr`` (CUDA build only, same arguments): one spline of ``length`` knots,
  fitted in place.
- ``CubicSplineWrapCPU`` / ``CubicSplineWrapGPU`` (the backend's ``CubicSplineWrap``): built
  from ``(x0, y0, c1, c2, c3, ninterps, length, spline_type)``, i.e.
  :attr:`~gpubackendtools.interpolate.CubicSplineInterpolant.cpp_class_args`. Its
  ``eval_wrap(y_new, x_new, spline_index, N)`` evaluates ``N`` points into ``y_new``, with
  ``spline_index`` an int32 array naming the spline of each point. The wrapper keeps pointers
  to the input arrays (no copy), so the arrays must outlive it.
- ``CubicSplineCPU`` / ``CubicSplineGPU``: the underlying C++ class, registered for type checks
  only (no Python constructor).
- ``CUBIC_SPLINE_LINEAR_SPACING``, ``CUBIC_SPLINE_LOG10_SPACING``,
  ``CUBIC_SPLINE_GENERAL_SPACING``: the spacing codes.

JAX spline functions
--------------------

Used by :class:`~gpubackendtools.interpolate.CubicSplineInterpolant` on the ``gbt_jax``
backend. The module imports ``jax`` at import time.

.. automodule:: gpubackendtools.jax.cubic_spline

.. autofunction:: gpubackendtools.jax.cubic_spline.fit_cubic_spline_not_a_knot

.. autofunction:: gpubackendtools.jax.cubic_spline.interpolate_wrap_jax

.. autofunction:: gpubackendtools.jax.cubic_spline.evaluate_cubic_spline

.. autoclass:: gpubackendtools.jax.cubic_spline.CubicSplineWrapJAX
    :members:
