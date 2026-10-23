"""Small device helpers (CuPy optional)."""

import warnings


try:
    from cupy.cuda.runtime import setDevice


except (ModuleNotFoundError, ImportError):
    setDevice = None


# TODO check: keep/remove/document -- no callers; superseded by _CudaBackend.set_cuda_device
def cuda_set_device(dev):
    """Globally sets CUDA device

    Calls ``cupy.cuda.runtime.setDevice`` (current device of the calling host thread); if
    CuPy could not be imported, only warns.

    Args:
        dev (int): CUDA device number.

    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")
