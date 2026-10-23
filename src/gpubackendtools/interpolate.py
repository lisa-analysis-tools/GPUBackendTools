# Interpolant for GPUs

# Copyright (C) 2021 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Batched cubic-spline interpolation, the reference ``gpubackendtools`` compute object.

:class:`CubicSplineInterpolant` fits many independent not-a-knot cubic splines at
construction (native ``interpolate_wrap`` kernel on CPU / CUDA, scipy on the JAX backend)
and evaluates them with array code on the backend's ``xp``. Downstream packages also pass
its flat coefficient arrays to native code (:attr:`CubicSplineInterpolant.container`,
:attr:`CubicSplineInterpolant.cpp_class_args`). :func:`searchsorted2d_vec` is the row-wise
``searchsorted`` used by the evaluation.
"""

import numpy as np

import warnings
from .parallelbase import GBTParallelModuleBase
from .pointeradjust import wrapper


def searchsorted2d_vec(a, b, xp=None, **kwargs):
    """Row-wise ``searchsorted`` of ``b`` into ``a`` in a single vectorized call.

    Each row is shifted by ``row_index * M``, with ``M`` one more than the larger of the
    value spans of ``a`` and ``b``, and one flat ``xp.searchsorted`` is run; the shift is
    then removed from the indices. The result is exact when the shifted rows do not
    interleave, i.e. when ``a`` and ``b`` cover a common value range (queries inside the
    knot range, as in :meth:`CubicSplineInterpolant.__call__`). On CuPy the device is
    synchronized before returning.

    Args:
        a (xp.ndarray): Sorted rows, shape ``(m, n)``.
        b (xp.ndarray): Query rows, shape ``(m, k)``.
        xp (module, optional): Array module (``numpy`` or ``cupy``). Default ``None``
            means ``numpy``.
        **kwargs: Forwarded to ``xp.searchsorted`` (e.g. ``side="right"``).

    Returns:
        xp.ndarray: Integer insertion indices into each row of ``a``, shape ``(m, k)``.

    Example:
        >>> import numpy as np
        >>> from gpubackendtools.interpolate import searchsorted2d_vec
        >>> a = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        >>> b = np.array([[0.5, 1.5], [1.0, 1.9]])
        >>> searchsorted2d_vec(a, b, side="right").tolist()
        [[1, 2], [2, 2]]
    """
    if xp is None:
        xp = np

    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)

    out = p - n * (xp.arange(m)[:, None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out


# Knot-spacing codes, same values as the CUBIC_SPLINE_* constants of InterpolateDevice.hh
# (also exported by the native ``interp`` modules); passed to CubicSplineWrap as spline_type.
CUBIC_SPLINE_LINEAR_SPACING = 1
CUBIC_SPLINE_LOG10_SPACING = 2
CUBIC_SPLINE_GENERAL_SPACING = 3 


class CubicSplineInterpolant(GBTParallelModuleBase):
    r"""GPU-accelerated Multiple Cubic Splines.

    This class produces multiple cubic splines. The cubic splines are produced
    with "not-a-knot" boundary conditions (the same splines as
    ``scipy.interpolate.CubicSpline`` with its default boundary condition).

    This class has GPU capability. The fit runs once, at construction: on the C++ backends
    through the native ``interpolate_wrap`` (a tridiagonal solve with LAPACKE ``dgtsv`` on
    CPU, cuSPARSE ``gtsv2StridedBatch`` on GPU), which fills the coefficient buffers in
    place; on ``gbt_jax`` through :func:`gpubackendtools.jax.cubic_spline.interpolate_wrap_jax`
    (a host-side scipy fit). Evaluation (:meth:`__call__`) is array code on :attr:`xp`.
    On segment ``i`` the spline is

    .. math::

        y(x) = y_i + c_{1,i} (x - x_i) + c_{2,i} (x - x_i)^2 + c_{3,i} (x - x_i)^3 .

    The knot spacing is detected from ``x`` (linear, log10 or general) and stored in
    :attr:`spline_type`; native consumers use it to pick the segment search.

    Inputs are converted to the backend's array module with ``xp.asarray``, so NumPy input
    to a CUDA backend is transferred to the device; ``x`` is always copied, while a 1D
    ``y_all`` already on the right module is stored without a copy. The native kernels
    require float64.

    Args:
        x (xp.ndarray): Independent-variable values for the splines. Either a
            1D flattened array of total length ``(ninterps * length)`` or an
            N-D array whose **last** axis is the per-spline ``length`` and whose
            leading axes enumerate the independent splines (so
            ``ninterps = prod(x.shape[:-1])``).
        y_all (xp.ndarray): Dependent-variable values, **the same shape as**
            ``x``.
        ninterps (int, optional): Number of independent splines. Required only
            when ``x`` / ``y_all`` are passed flattened (1D); otherwise it is
            inferred from ``x.shape[:-1]``. (Default: ``None``)
        length (int, optional): Number of points per spline. Required only when
            ``x`` / ``y_all`` are passed flattened (1D); otherwise it is
            ``x.shape[-1]``. (Default: ``None``)
        spline_type (int, optional): Ignored: the spacing type is always
            auto-detected from ``x`` and this value is overwritten. (Default:
            ``None``)
        force_backend (str | Backend | tuple[str, str], optional): ``"cpu"``,
            ``"cuda11x"``, ``"cuda12x"``, ``"cuda13x"``, ``"jax"``, or the aliases
            ``"cuda"`` / ``"gpu"`` (first available CUDA flavor); ``"gbt_"`` is
            prepended unless the string already starts with it (see
            :class:`~gpubackendtools.parallelbase.GBTParallelModuleBase`).
            Chooses the compute backend at construction. (Default: ``None`` -- first
            available of ``gbt_cuda13x``, ``gbt_cuda12x``, ``gbt_cuda11x``,
            ``gbt_cpu``, ``gbt_jax``.)

    Raises:
        ValueError: If input arguments are not correct: only one of ``x`` / ``y_all``
            flattened, flattened input without ``length`` and ``ninterps``, lengths that
            do not equal ``length * ninterps``, or N-D inputs of different shapes.

    Attributes:
        length (int): Number of knots per spline.
        ninterps (int): Number of splines.
        reshape_shape (tuple[int, ...]): Shape of the input ``x`` (N-D input) or
            ``(ninterps, length)`` (flattened input).
        degree (int): Always ``3``.
        x_flat (xp.ndarray): Knots, shape ``(ninterps * length,)``.
        y_flat (xp.ndarray): Values at the knots, shape ``(ninterps * length,)``.
        c1_flat (xp.ndarray): Linear coefficients, shape ``(ninterps * length,)``; the
            last entry of each spline is not used by the evaluation.
        c2_flat (xp.ndarray): Quadratic coefficients, same layout.
        c3_flat (xp.ndarray): Cubic coefficients, same layout.

    Example:
        >>> import numpy as np
        >>> from gpubackendtools.interpolate import CubicSplineInterpolant
        >>> x = np.tile(np.linspace(0.0, 1.0, 100), (2, 1))  # 2 splines of 100 knots
        >>> y = np.stack([np.sin(2 * np.pi * x[0]), np.cos(2 * np.pi * x[1])])
        >>> spl = CubicSplineInterpolant(x, y, force_backend="cpu")
        >>> x_new = np.tile(np.linspace(0.1, 0.9, 7), (2, 1))
        >>> spl(x_new).shape
        (2, 7)
        >>> spl(np.array([[0.25, 0.5]]), ind_interps=np.array([1])).shape  # spline 1 only
        (1, 2)
    """

    def __init__(
        self,
        x,
        y_all,
        ninterps=None,
        length=None,
        spline_type=None,
        force_backend=None,
    ):

        # check all inputs

        super().__init__(force_backend=force_backend)

        # first check is for flattened arrays
        if x.ndim == 1 or y_all.ndim == 1:
            if x.ndim != 1 or y_all.ndim != 1:
                raise ValueError(
                    "If providing flattened x and y_all, need to both be flattened."
                )
            if len(x) != len(y_all):
                raise ValueError("x and y must have same length.")
            
            if (
                length is None
                or ninterps is None
            ):
                raise ValueError(
                    "If providing flattened arrays, need to provide dimensional information: length, ninterps."
                )

            if len(x) != length * ninterps:
                raise ValueError(
                    f"Length of the x array is not correct. It is supposed to be {length * ninterps}. It is currently {len(x)}."
                )
            if len(y_all) != length * ninterps:
                raise ValueError(
                    f"Length of the y_all array is not correct. It is supposed to be {length * ninterps}. It is currently {len(y_all)}."
                )
            self.length = length
            self.ninterps = ninterps
            self.reshape_shape = (self.ninterps, self.length)

        else:
            # assumes last dimension is length
            if x.shape != y_all.shape:
                raise ValueError("x and y must have the same shape with the final dimension being the length of the spline.")
            
            # arrays are shaped
            self.reshape_shape = x.shape

            self.length = x.shape[-1]
            self.ninterps = int(np.prod(x.shape[:-1]))
            x = x.flatten()
            y_all = y_all.flatten()

        # get/store info
        self.degree = 3

        # setup all arrays for interpolation -- coerce inputs onto THIS
        # backend's xp so callers may pass numpy/cupy (or jax) interchangeably.
        # The native interpolate_wrap requires arrays matching the backend
        # device; previously self.x_flat/self.y_flat were stored UNCOERCED (the
        # asarray result here was computed and then dropped), so numpy inputs to
        # a CUDA backend hit a nanobind device mismatch.
        x = self.xp.asarray(x)
        y_all = self.xp.asarray(y_all)
        B_flat = self.xp.zeros((self.ninterps * self.length,))
        self.y_flat = y_all
        self.x_flat = x.copy() if hasattr(x, "copy") else x

        if self.xp.allclose((_diff := self.xp.diff(self.x, axis=-1)), _diff[..., 0][..., None]):
            spline_type = CUBIC_SPLINE_LINEAR_SPACING
        elif self.xp.allclose((_diff := self.xp.diff(self.xp.log10(self.x), axis=-1)), _diff[..., 0][..., None]):
            spline_type = CUBIC_SPLINE_LOG10_SPACING
        else:
            spline_type = CUBIC_SPLINE_GENERAL_SPACING

        self.spline_type = spline_type

        if spline_type == CUBIC_SPLINE_LINEAR_SPACING:
            assert self.xp.allclose(self.xp.diff(self.x_interp_shape, axis=-1), self.xp.diff(self.x_interp_shape, axis=-1)[:, 0][:, None])

        elif spline_type == CUBIC_SPLINE_LOG10_SPACING:
            assert self.xp.allclose(self.xp.diff(self.xp.log10(self.x_interp_shape), axis=-1), self.xp.diff(self.xp.log10(self.x_interp_shape), axis=-1)[:, 0][:, None])

        # Backend split: the C++ backends use in-place mutation of
        # c1/c2/c3 buffers (aliased with upper/diag/lower of the
        # tridiagonal system). The JAX backend can't mutate -- its
        # ``interpolate_wrap`` returns the fitted coefficients
        # functionally instead.
        if self.backend.name == "gbt_jax":
            c1_flat, c2_flat, c3_flat = self.interpolate_arrays(
                self.x_flat, self.y_flat, B_flat,
                B_flat, B_flat, B_flat,    # the three buffer slots are unused on JAX
                self.length, self.ninterps,
            )
            self.c1_flat = c1_flat
            self.c2_flat = c2_flat
            self.c3_flat = c3_flat
        else:
            self.c1_flat = upper_diag = self.xp.zeros_like(B_flat)
            self.c2_flat = diag = self.xp.zeros_like(B_flat)
            self.c3_flat = lower_diag = self.xp.zeros_like(B_flat)
            self.interpolate_arrays(
                self.x_flat,
                self.y_flat,
                B_flat,
                upper_diag,
                diag,
                lower_diag,
                self.length,
                self.ninterps,
            )

    @property
    def spline_type(self) -> int:
        """Detected knot spacing code: 1 linear, 2 log10, 3 general.

        The values are the module constants ``CUBIC_SPLINE_LINEAR_SPACING``,
        ``CUBIC_SPLINE_LOG10_SPACING`` and ``CUBIC_SPLINE_GENERAL_SPACING``.

        Raises:
            ValueError: On assignment of any other value.
        """
        return self._spline_type
    
    @spline_type.setter
    def spline_type(self, spline_type: int):
        if spline_type not in [CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING]:
            raise ValueError("spline_type must be one of CUBIC_SPLINE_LINEAR_SPACING, CUBIC_SPLINE_LOG10_SPACING, CUBIC_SPLINE_GENERAL_SPACING.")
        self._spline_type = spline_type

    @property
    def xp(self) -> object:
        """Array module of the backend: ``numpy``, ``cupy`` or ``jax.numpy``."""
        return self.backend.xp
    
    @classmethod
    def supported_backends(cls) -> list:
        """``gbt_cuda13x``, ``gbt_cuda12x``, ``gbt_cuda11x``, ``gbt_cpu``, then ``gbt_jax``."""
        # Append the pure-JAX backend after the GPU/CPU options so that
        # the default "first available" pick stays GPU/CPU when both
        # are present. Users opt into the JAX path with
        # force_backend="jax" (or by passing in JAX arrays via
        # downstream code that already resolved to gbt_jax).
        return ["gbt_" + _tmp for _tmp in cls.GPU_RECOMMENDED()] + ["gbt_jax"]

    @property
    def interpolate_arrays(self) -> callable:
        """C/CUDA wrapped function for computing interpolation (``backend.interpolate_wrap``).

        Signature ``(x, y, B, upper_diag, diag, lower_diag, length, ninterps)`` on flat
        float64 arrays of size ``length * ninterps``. The native version fills the last
        three buffers in place with ``c1``, ``c2``, ``c3``; the JAX version returns
        ``(c1, c2, c3)``.
        """
        return self.backend.interpolate_wrap

    @property
    def x(self):
        """Get shaped x array: knots with the input shape (:attr:`reshape_shape`)."""
        return self.x_flat.reshape(self.reshape_shape)
    
    @property
    def x_interp_shape(self):
        """Get x array with shape ``(ninterps, length)``."""
        return self.x_flat.reshape(self.ninterps, self.length)

    @property
    def y(self):
        """Get shaped y array (input shape, :attr:`reshape_shape`)."""
        return self.y_flat.reshape(self.reshape_shape)
    
    @property
    def y_interp_shape(self):
        """Get y array with shape ``(ninterps, length)``."""
        return self.y_flat.reshape(self.ninterps, self.length)

    @property
    def c1(self):
        """Get shaped c1 (linear coefficient) array (input shape, :attr:`reshape_shape`)."""
        return self.c1_flat.reshape(self.reshape_shape)

    @property
    def c1_interp_shape(self):
        """Get c1 array with shape ``(ninterps, length)``."""
        return self.c1_flat.reshape(self.ninterps, self.length)
    
    @property
    def c2(self):
        """Get shaped c2 (quadratic coefficient) array (input shape, :attr:`reshape_shape`)."""
        return self.c2_flat.reshape(self.reshape_shape)

    @property
    def c2_interp_shape(self):
        """Get c2 array with shape ``(ninterps, length)``."""
        return self.c2_flat.reshape(self.ninterps, self.length)
    
    @property
    def c3(self):
        """Get shaped c3 (cubic coefficient) array (input shape, :attr:`reshape_shape`)."""
        return self.c3_flat.reshape(self.reshape_shape)

    @property
    def c3_interp_shape(self):
        """Get c3 array with shape ``(ninterps, length)``."""
        return self.c3_flat.reshape(self.ninterps, self.length)
    
    @property
    def container(self):
        """Container for easy transit of interpolation information.

        Returns:
            list[xp.ndarray]: ``[x_flat, y_flat, c1_flat, c2_flat, c3_flat]``, each of shape
            ``(ninterps * length,)`` (references, not copies).
        """
        return [self.x_flat, self.y_flat, self.c1_flat, self.c2_flat, self.c3_flat]
    
    @property
    def cpp_class_args(self) -> tuple:
        """Constructor arguments of the native ``backend.CubicSplineWrap``.

        Returns:
            tuple: ``(x_flat, y_flat, c1_flat, c2_flat, c3_flat, ninterps, length,
            spline_type)``.
        """
        return (
            self.x_flat, 
            self.y_flat, 
            self.c1_flat, 
            self.c2_flat, 
            self.c3_flat, 
            self.ninterps, 
            self.length, 
            self.spline_type
        )

    @property
    def cpp_class(self):
        """Native ``CubicSplineWrap`` over this object's coefficient arrays.

        Each access constructs a new wrapper from :attr:`cpp_class_args` and keeps it on
        ``self._cpp_class``. The native wrapper's ``eval_wrap(y_new, x_new, spline_index, N)``
        evaluates ``N`` points on the backend's device (``spline_index`` int32, one spline
        index per point). On ``gbt_jax`` this is a
        :class:`~gpubackendtools.jax.cubic_spline.CubicSplineWrapJAX`, which has no
        ``eval_wrap``.
        """
        # must store it or will lose access to attributes (do not do return self.backend...)
        self._cpp_class = self.backend.CubicSplineWrap(*self.cpp_class_args)
        return self._cpp_class
    
    def __call__(self, x_new, ind_interps = None, use_c_backend=False, error_out_of_bounds = True, derivative=0):
        """Evaluate the fitted splines at new abscissae.

        Segments are located with :func:`searchsorted2d_vec` (``side="right"``) and the
        segment cubic (or its derivative) is evaluated with :attr:`xp` array code; the
        native ``CubicSpline`` evaluator is not used. On CuPy the default memory pool is
        freed (``free_all_blocks``) before returning.

        Args:
            x_new (xp.ndarray): Query points, on the backend's array module. When
                ``ind_interps`` is ``None``, its leading axes must match ``x.shape[:-1]``
                (one query row per spline; ``(ninterps, n_new)`` for flattened input);
                the last axis is the queries for that spline.
            ind_interps (xp.ndarray, optional): Integer indices selecting which
                splines to evaluate (must be unique and in ``[0, ninterps)``), with shape
                ``x_new.shape[:-1]``. Required when ``x_new`` does not have one row per
                spline. (Default: ``None``.)
            use_c_backend (bool, optional): Reserved; currently raises
                ``NotImplementedError`` if ``True``. (Default: ``False``.)
            error_out_of_bounds (bool, optional): If ``True``, raise when any
                ``x_new`` falls outside the fitted range of its spline; if ``False``,
                warn and set the out-of-range outputs to ``0.0``. (Default: ``True``.)
            derivative (int, optional): Derivative order to return, ``0`` (the
                interpolant itself) to ``3``. (Default: ``0``.)

        Returns:
            xp.ndarray: Interpolated values (or derivatives) with the shape of ``x_new``.

        Raises:
            ValueError: If ``x_new`` / ``ind_interps`` shapes do not match, a point is out
                of bounds with ``error_out_of_bounds=True``, or ``derivative`` is not 0-3.
            AssertionError: If ``ind_interps`` is out of range or has duplicates.
            NotImplementedError: If ``use_c_backend`` is ``True``.

        Warning:
            On ``gbt_jax`` the in-place index fixes (a query equal to the last knot, or
            zeroing out-of-range outputs) assign into immutable JAX arrays and fail.
        """

        if use_c_backend:
            raise NotImplementedError
        
        input_shape = x_new.shape
        if ind_interps is None:
            if not x_new.shape[:-1] == self.reshape_shape[:-1]:
                raise ValueError("Must add ind_interps if x_new is not same shape (except for the last axis) as the input x array.")
            
            ind_interps = self.xp.arange(self.ninterps)
            
        else:
            assert ind_interps.max().item() < self.ninterps
            assert ind_interps.min().item() >= 0
            if ind_interps.shape != x_new.shape[:-1]:
                raise ValueError("When inputing ind_interps, the shape needs to match x_new.shape[:-1].")
            ind_interps = ind_interps.flatten()

        assert len(ind_interps) == len(self.xp.unique(ind_interps))

        num_interps_here = len(ind_interps)

        ind_interps_all = self.xp.repeat(ind_interps[:, None], x_new.shape[-1], axis=-1).reshape(num_interps_here, x_new.shape[-1])
            
        x_new = x_new.reshape(num_interps_here, x_new.shape[-1])

        assert x_new.shape == ind_interps_all.shape
        bool1 = x_new <= self.x_interp_shape[ind_interps].max(axis=-1)[:, None]
        bool2 = x_new >= self.x_interp_shape[ind_interps].min(axis=-1)[:, None]
        fix = False
        if not (
            self.xp.all(
                bool1
            ) and self.xp.all(
                bool2
            )
        ):
            if error_out_of_bounds:
                raise ValueError("New x array values are not within the bounds of the input x array for the spline. Either change the new xarray or run with error_out_of_bounds = False.") 
            else:
                fix = True
                
        segment_inds = (
            searchsorted2d_vec(
                self.x_interp_shape[ind_interps],
                x_new.reshape(num_interps_here, x_new.shape[-1]),
                xp=self.xp,
                side="right",
            )
            - 1
        ).reshape(x_new.shape)

        if self.xp.any(segment_inds == self.length - 1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == self.length - 1] = self.length - 2

        x0 = self.x.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        y0 = self.y.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]

        c1 = self.c1.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        c2 = self.c2.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]
        c3 = self.c3.reshape(self.ninterps, self.length)[ind_interps_all, segment_inds]

        dx = x_new - x0

        if derivative == 0:
            y_new = y0 + c1 * dx + c2 * dx**2 + c3 * dx**3
        elif derivative == 1:
            y_new = c1 + 2 * c2 * dx + 3 * c3 * dx**2
        elif derivative == 2:
            y_new = 2 * c2 + 6 * c3 * dx
        elif derivative == 3:
            y_new = 6 * c3
        else:
            raise ValueError("Invalid derivative order.")

        if fix:
            warnings.warn("New x array contains values outside domain of spline. Putting zeros outside domain.")
            y_new[~(bool1 & bool2)] = 0.0

        if hasattr(self.xp, "cuda"):
            self.xp.get_default_memory_pool().free_all_blocks()
        return y_new.reshape(input_shape)
    
    # TODO check: keep/remove/document -- no callers; uses undefined x_shaped/y_shaped/c*_shaped
    def interp_special(self, x_new, inds):
        assert x_new.ndim == 1
        assert x_new.shape[0] == inds.shape[0]

        assert self.xp.all(
            x_new <= self.x_shaped.reshape(-1, self.length)[inds].max(axis=-1)
        ) and self.xp.all(x_new >= self.x_shaped.reshape(-1, self.length)[inds].min(axis=-1))

        max_x = self.x_shaped.reshape(-1, self.length)[inds].max().item()
        scaled_x = ((self.x_shaped.reshape(-1, self.length)[np.unique(inds)] / max_x) + 100 * np.unique(inds)[:, None]).flatten()
        scaled_x_new = x_new / max_x + 100 * inds

        segment_inds = self.xp.searchsorted(scaled_x, scaled_x_new, side="right") - 1

        if self.xp.any(segment_inds == -1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == -1] = 0

        if self.xp.any(segment_inds == self.length - 1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == self.length - 1] = self.length - 2

        x0 = self.x_shaped.reshape(-1, self.length)[(inds, segment_inds)]
        
        _inds_y0 = self.xp.repeat(inds, self.y_shaped.shape[0])
        _segment_inds_y0 = self.xp.repeat(segment_inds, self.y_shaped.shape[0])
        _interp_params_inds = self.xp.tile(self.xp.arange(self.y_shaped.shape[0]), (len(inds),))
        y0 = self.y_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c1 = self.c1_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c2 = self.c2_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)
        c3 = self.c3_shaped.reshape(self.y_shaped.shape[0], -1, self.length)[_interp_params_inds, _inds_y0, _segment_inds_y0].reshape(self.y_shaped.shape[0], -1)

        dx = x_new - x0
        y_new = y0 + c1 * dx[None, :] + c2 * dx[None, :] **2 + c3 * dx[None, :]**3

        if hasattr(self.xp, "cuda"):
            self.xp.get_default_memory_pool().free_all_block()
        return y_new
