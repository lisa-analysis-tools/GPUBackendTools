Parallel Module Base Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every class of the stack that dispatches to a backend derives from
:class:`~gpubackendtools.parallelbase.ParallelModuleBase`, directly or through a package
base (``lisatools.utils.parallelbase.LISAToolsParallelModule``,
``gbgpu.parallelbase.GBGPUParallelModule``, ``bbhx.utils.parallelbase.BBHxParallelModule``,
``few.utils.baseclasses.FewBackendConsumer``). The backend is chosen once, at construction,
with ``force_backend``, and the instance keeps only its registry name, so it deepcopies and
pickles cleanly; :attr:`~gpubackendtools.parallelbase.ParallelModuleBase.backend` and
:attr:`~gpubackendtools.parallelbase.ParallelModuleBase.xp` look the backend up again on each
access.

``force_backend`` forms accepted by :class:`~gpubackendtools.parallelbase.ParallelModuleBase`:

.. list-table::
   :header-rows: 1

   * - Value
     - Resolved with
   * - ``None``
     - :func:`~gpubackendtools.globals.get_first_backend` on
       :meth:`~gpubackendtools.parallelbase.ParallelModuleBase.supported_backends`
   * - ``str``
     - :func:`~gpubackendtools.globals.get_backend`: a full registry name (``"gbt_cpu"``) or
       a ``"<pkg>_cuda"`` / ``"<pkg>_gpu"`` alias
   * - :class:`~gpubackendtools.gpubackendtools.Backend`
     - its registry :attr:`~gpubackendtools.gpubackendtools.Backend.name`
   * - ``(package, flavor)`` tuple or list
     - joined as ``"package_flavor"``

Bare flavors (``"cpu"``, ``"cuda12x"``, ``"cuda"``, ``"gpu"``, ``"jax"``) work only through a
class that adds its package prefix: :class:`~gpubackendtools.parallelbase.GBTParallelModuleBase`
for the ``gbt_*`` backends (used by
:class:`~gpubackendtools.interpolate.CubicSplineInterpolant`), and the package bases above for
theirs. Passing an existing backend's
:attr:`~gpubackendtools.gpubackendtools.Backend.backend_base` (its flavor, e.g. ``"cuda12x"``)
to such a class selects the same device in that package.

.. automodule:: gpubackendtools.parallelbase

.. autoclass:: gpubackendtools.parallelbase.ParallelModuleBase
    :members:
    :show-inheritance:
    :exclude-members: ParallelModuleDerivate

.. autoclass:: gpubackendtools.parallelbase.GBTParallelModuleBase
    :show-inheritance:
