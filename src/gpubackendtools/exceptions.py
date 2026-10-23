"""Definition of GPUBACKENDTOOLS package common exceptions.

:class:`GPUBACKENDTOOLSException` is the root of every ``gpubackendtools`` exception
(catching it also catches the backend-loading exceptions of
:mod:`gpubackendtools.gpubackendtools`). ``ExceptionGroup`` is re-exported (the builtin on
Python >= 3.11, the ``exceptiongroup`` backport otherwise).

Note:
    :class:`BackendUnavailableException` and :class:`BackendNotInstalled` here are NOT the
    classes of the same name in :mod:`gpubackendtools.gpubackendtools`. The backend
    registry only catches the :mod:`gpubackendtools.gpubackendtools` ones when it marks a
    backend unavailable; the ones defined here propagate out of
    :func:`~gpubackendtools.globals.get_backend` / ``has_backend``.
"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    ExceptionGroup = ExceptionGroup


class GPUBACKENDTOOLSException(Exception):
    """Base class for GPUBACKENDTOOLS package exceptions."""

    pass


class BackendUnavailableException(GPUBACKENDTOOLSException):
    """Exception raised when the backend is not available.

    Not caught by the backend registry; see the module note and
    :class:`gpubackendtools.gpubackendtools.BackendUnavailableException`.
    """

    pass


class BackendNotInstalled(BackendUnavailableException):
    """Exception raised when the backend has not been installed"""
    pass


class CudaException(GPUBACKENDTOOLSException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(GPUBACKENDTOOLSException):
    """Base class for CuPy-related exceptions."""

    pass


class MissingDependency(GPUBACKENDTOOLSException):
    """Exception raised when a required dependency is missing.

    A plain exception taking only positional message arguments (unlike
    :class:`gpubackendtools.gpubackendtools.MissingDependencies`, it accepts no
    ``pip_deps`` / ``conda_deps``).
    """

    pass
