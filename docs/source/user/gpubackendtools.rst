Backends and the Registry
~~~~~~~~~~~~~~~~~~~~~~~~~

A *backend* is one compute flavor of one package: an array module ``xp`` (``numpy``,
``cupy`` or ``jax.numpy``) plus that package's compiled (or pure-Python) callables for the
device. Every package of the stack registers its backend classes on one shared registry,
``Globals().backends_manager`` (a :class:`~gpubackendtools.gpubackendtools.BackendsManager`),
when it is imported. A backend is constructed, and its plugin module imported, on its first
lookup; the instance is then shared by the whole process.

Backend names
-------------

Every backend class sets two names as class attributes:

- ``_name``: the **registry key** ``"<pkg>_<flavor>"``, exposed as
  :attr:`~gpubackendtools.gpubackendtools.Backend.name` (``"gbt_cpu"``,
  ``"lisatools_cuda12x"``, ...). The lookup functions and pickling use this name.
- ``_backend_name``: the importable **plugin module** ``"<pkg>_backend_<flavor>"``, exposed
  as :attr:`~gpubackendtools.gpubackendtools.Backend.backend_name` (``"gbt_backend_cpu"``).
  The CPU and CUDA base classes import it to check that the plugin is installed.

The flavors are ``cpu``, ``cuda11x``, ``cuda12x``, ``cuda13x`` and ``jax``.
``import gpubackendtools`` registers the package's own backends:

.. list-table::
   :header-rows: 1

   * - Registry key
     - Plugin module
     - ``xp``
     - Notes
   * - ``gbt_cpu``
     - ``gbt_backend_cpu``
     - ``numpy``
     - Feature ``NUMPY``.
   * - ``gbt_cuda11x``, ``gbt_cuda12x``, ``gbt_cuda13x``
     - ``gbt_backend_cuda11x``, ``gbt_backend_cuda12x``, ``gbt_backend_cuda13x``
     - ``cupy``
     - Features ``CUPY | CUDA | GPU``. Need CuPy built for the same CUDA major version;
       supported driver CUDA ranges ``[11.2, 12.0)``, ``[12.0, 13.0)``, ``[13.0, 14.0)``
       (the driver version is checked only when the first load attempt fails).
       For ``cuda11x`` see the warning on
       :class:`~gpubackendtools.gpubackendtools.Cuda11xBackend`.
   * - ``gbt_jax``
     - none (pure Python)
     - ``jax.numpy``
     - Registered only when ``import jax`` works. Feature ``NUMPY`` (``uses_gpu`` is
       ``False``). Loading it turns on ``jax_enable_x64`` for the process.

Lookup
------

- :func:`~gpubackendtools.globals.get_backend` takes a full registry key, or the aliases
  ``"<pkg>_cuda"`` (the first available of ``<pkg>_cuda13x``, ``<pkg>_cuda12x``,
  ``<pkg>_cuda11x``) and ``"<pkg>_gpu"`` (same as ``"<pkg>_cuda"``). A bare flavor such as
  ``"cpu"`` is not a registry key and raises ``ValueError``.
- :func:`~gpubackendtools.globals.has_backend` takes a full registry key only. It returns
  ``False`` for a disabled or unavailable backend; the ``cuda`` / ``gpu`` aliases, bare or
  prefixed, raise ``ValueError``.
- :func:`~gpubackendtools.globals.get_first_backend` takes a preference list. Each entry is a
  full registry key or a bare flavor; a bare flavor matches the first registered key, in
  registration order, whose part after the last ``_`` equals it. ``gbt_*`` is registered
  first, so bare flavors resolve to ``gpubackendtools``' own backends: other packages pass
  prefixed names.
- Compute objects pick their backend once, at construction, with ``force_backend``; see
  :doc:`parallelbase`.

The three functions are also importable from the top-level package
(``from gpubackendtools import get_backend``).

``copy.copy`` and ``copy.deepcopy`` of a backend return the same instance, and pickling
stores only the registry key, which unpickling resolves in the receiving process (if the key
is unknown there, the package named by the part before the first ``_`` is imported first).
Every :meth:`~gpubackendtools.gpubackendtools.BackendsManager.add_backends` call rebuilds the
registry for all packages, so a lookup after a later registration builds a new instance.

Writing a downstream backend
----------------------------

A package ``mypkg`` adds its backends the way :mod:`gpubackendtools.cutils` does for
``gpubackendtools`` itself; LISAanalysistools, GBGPU, BBHx and FastEMRIWaveforms follow the
same pattern.

1. **Method table.** Subclass :class:`~gpubackendtools.gpubackendtools.BackendMethods` (a
   dataclass whose only field is ``xp``) with one field per native symbol, and write a mixin
   that copies those fields onto the backend instance (see
   :class:`~gpubackendtools.cutils.GBTBackendMethods` and
   :class:`~gpubackendtools.cutils.GBTBackend`).
2. **One class per flavor.** Derive from the device base
   (:class:`~gpubackendtools.gpubackendtools.CpuBackend`,
   :class:`~gpubackendtools.gpubackendtools.Cuda11xBackend`,
   :class:`~gpubackendtools.gpubackendtools.Cuda12xBackend` or
   :class:`~gpubackendtools.gpubackendtools.Cuda13xBackend`) and the mixin. Set
   ``_name = "mypkg_<flavor>"`` and ``_backend_name = "mypkg_backend_<flavor>"``; ``mypkg``
   must be the importable package name, because unpickling imports it, and must contain no
   ``_``, because the lookups and unpickling split registry keys on ``_``.
3. **Loader.** Implement the device base's loader static method (``cpu_methods_loader``,
   ``cuda11x_module_loader``, ``cuda12x_module_loader`` or ``cuda13x_module_loader``): import
   the compiled extension from the plugin module and return the method table with ``xp`` set
   (``numpy`` or ``cupy``). When it cannot load, raise
   :class:`gpubackendtools.gpubackendtools.BackendUnavailableException` or a subclass
   (:class:`~gpubackendtools.gpubackendtools.MissingDependencies`, ...): the registry then
   records the backend as unavailable (``has_backend`` returns ``False``,
   ``get_first_backend`` tries the next candidate). Any other exception propagates out of
   the lookup. For ``cuda11x`` the device base currently calls its own abstract loader
   instead of the override (see the warning on
   :class:`~gpubackendtools.gpubackendtools.Cuda11xBackend`).
4. **Registration.** At package import, call
   ``Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend, ...})``.
5. **Compute objects.** Derive the package's base class from
   :class:`~gpubackendtools.parallelbase.ParallelModuleBase`, prefix bare ``force_backend``
   strings with ``"mypkg_"`` (copy :class:`~gpubackendtools.parallelbase.GBTParallelModuleBase`
   with your own ``_BACKEND_PREFIX``), and return prefixed names from ``supported_backends``.

.. code-block:: python

    import dataclasses
    import typing

    from gpubackendtools import Globals
    from gpubackendtools.gpubackendtools import (
        BackendMethods,
        BackendUnavailableException,
        CpuBackend,
    )


    @dataclasses.dataclass
    class MyPkgBackendMethods(BackendMethods):
        compute: typing.Callable


    class MyPkgBackend:
        def __init__(self, methods: MyPkgBackendMethods):
            self.compute = methods.compute


    class MyPkgCpuBackend(CpuBackend, MyPkgBackend):
        _name = "mypkg_cpu"
        _backend_name = "mypkg_backend_cpu"

        def __init__(self):
            CpuBackend.__init__(self)
            MyPkgBackend.__init__(self, self.cpu_methods_loader())

        @staticmethod
        def cpu_methods_loader() -> MyPkgBackendMethods:
            try:
                import mypkg_backend_cpu.ext
            except ImportError as e:
                raise BackendUnavailableException("'cpu' backend could not be imported.") from e
            import numpy

            return MyPkgBackendMethods(xp=numpy, compute=mypkg_backend_cpu.ext.compute)


    Globals().backends_manager.add_backends({"mypkg_cpu": MyPkgCpuBackend})

On the build side, ``cmake_functions.cmake`` in the directory returned by
:func:`gpubackendtools.get_include` provides
``apply_cpu_backend_common_options(<libname> <pkg> <prefix> <is_static>)`` and
``apply_gpu_backend_common_options(...)`` (same arguments). They configure the existing
target ``<prefix>_cpu_<libname>`` (``<prefix>_gpu_<libname>``; with a ``_static`` suffix
when ``<is_static>`` is true), set its output name to ``<libname>`` and install it into the
plugin package ``<pkg>_backend_cpu`` (``<pkg>_backend_cuda<major>x``, with the major version
of the CUDA toolkit used for the build). For example, LISAanalysistools builds
``lisatools_backend_cpu.pycppdetector`` with
``apply_cpu_backend_common_options(pycppdetector lisatools lisaanalysistools false)``.

.. warning::

   :mod:`gpubackendtools.exceptions` defines a second, unrelated
   ``BackendUnavailableException``, which the registry does not catch. The loaders in
   :mod:`gpubackendtools.cutils`, and the downstream ``cutils`` modules that star-import
   :mod:`gpubackendtools.exceptions`, raise that class. So when a plugin package is
   installed but its extension fails to import, the lookup raises instead of reporting the
   backend unavailable. A missing plugin package is still reported correctly: the device
   bases import the plugin module themselves before calling the loader.

API reference
-------------

Package
^^^^^^^

.. automodule:: gpubackendtools

.. autofunction:: gpubackendtools.get_include

.. autofunction:: gpubackendtools.get_cmake_module_path

Backend classes
^^^^^^^^^^^^^^^

.. automodule:: gpubackendtools.gpubackendtools

.. autoclass:: gpubackendtools.gpubackendtools.Backend
    :members:
    :show-inheritance:
    :exclude-members: xp, features

.. autoclass:: gpubackendtools.gpubackendtools.BackendMethods
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.CpuBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools._CudaBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda11xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda12xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.Cuda13xBackend
    :members:
    :show-inheritance:

Registry
^^^^^^^^

.. autoclass:: gpubackendtools.gpubackendtools.BackendsManager
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendStatus

.. autoclass:: gpubackendtools.gpubackendtools.BackendStatusUnloaded
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendStatusLoaded
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendStatusDisabled
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendStatusUnavailable
    :show-inheritance:

Lookup functions
^^^^^^^^^^^^^^^^

.. autofunction:: gpubackendtools.globals.get_backend

.. autofunction:: gpubackendtools.globals.has_backend

.. autofunction:: gpubackendtools.globals.get_first_backend

GBT backends
^^^^^^^^^^^^

.. automodule:: gpubackendtools.cutils

.. autoclass:: gpubackendtools.cutils.GBTBackendMethods
    :show-inheritance:

.. autoclass:: gpubackendtools.cutils.GBTBackend

.. autoclass:: gpubackendtools.cutils.GBTCpuBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.cutils.GBTCuda11xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.cutils.GBTCuda12xBackend
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.cutils.GBTCuda13xBackend
    :members:
    :show-inheritance:

JAX backend
^^^^^^^^^^^

.. automodule:: gpubackendtools.jax

.. automodule:: gpubackendtools.jax.backend

.. autoclass:: gpubackendtools.jax.backend.GBTJaxBackend
    :show-inheritance:

The JAX spline functions are documented on :doc:`interpolate`.

Exceptions
^^^^^^^^^^

.. automodule:: gpubackendtools.exceptions

.. autoclass:: gpubackendtools.exceptions.GPUBACKENDTOOLSException
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.BackendUnavailableException
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.BackendNotInstalled
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.CudaException
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.CuPyException
    :show-inheritance:

.. autoclass:: gpubackendtools.exceptions.MissingDependency
    :show-inheritance:

Backend-loading exceptions defined in :mod:`gpubackendtools.gpubackendtools`; these are the
classes the registry catches when it marks a backend unavailable
(:class:`~gpubackendtools.gpubackendtools.BackendAccessException` is what the lookups raise
for a disabled or unavailable backend):

.. autoclass:: gpubackendtools.gpubackendtools.BackendUnavailableException
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendNotInstalled
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.MissingDependencies
    :members:
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.MissingDriver
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.SoftwareException
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.MissingHardware
    :show-inheritance:

.. autoclass:: gpubackendtools.gpubackendtools.BackendAccessException
    :show-inheritance:

Legacy pointer helpers
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: gpubackendtools.pointeradjust

.. autofunction:: gpubackendtools.pointeradjust.wrapper

.. autofunction:: gpubackendtools.pointeradjust.pointer_adjust
