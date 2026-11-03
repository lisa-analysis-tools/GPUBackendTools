"""Legacy helpers turning NumPy / CuPy arrays into raw pointers for Cython entry points.

:func:`wrapper` replaces array arguments by their data addresses (``int``) and objects with
a ``ptr`` attribute by that attribute; :func:`pointer_adjust` is the decorator form. The
active native path of the stack is nanobind, which takes arrays directly; these helpers
remain for Cython-era code. Whether CuPy arrays are recognized is decided once, at import
(module flag ``gpu``).
"""

# check to see if cupy is available for gpus
import numpy as np

try:
    import cupy as xp

    gpu = True

except (ImportError, ModuleNotFoundError):
    import numpy as np

    gpu = False


def wrapper(*args, **kwargs):
    """Function to convert array and C/C++ class arguments to ptrs.

    This function checks the object type. If it is a cupy or numpy array,
    it will determine its pointer by calling the proper attributes. If you design
    a Cython class to be passed through python, it must have a :code:`ptr`
    attribute.

    If you use this function, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    For a ``cupy.ndarray`` the value is ``arg.data.mem.ptr`` (the start of the underlying
    allocation, which differs from the array start for an offset view); for a
    ``numpy.ndarray`` it is ``arg.__array_interface__["data"][0]``. Other arguments pass
    through unchanged.

    Args:
        *args: Positional arguments of a function.
        **kwargs: Keyword arguments of a function.

    Returns:
        tuple[list, dict]: ``(targs, tkwargs)`` where t indicates target (with pointer
        values rather than python objects).

    Example:
        >>> import numpy as np
        >>> from gpubackendtools import wrapper
        >>> targs, tkwargs = wrapper(np.zeros(4), 3, flag=True)
        >>> isinstance(targs[0], int), targs[1], tkwargs
        (True, 3, {'flag': True})
    """
    # declare target containers
    targs = []
    tkwargs = {}

    # args first
    for arg in args:
        if gpu:
            # cupy arrays
            if isinstance(arg, xp.ndarray):
                targs.append(arg.data.mem.ptr)
                continue

        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if gpu:
            # cupy arrays
            if isinstance(arg, xp.ndarray):
                tkwargs[key] = arg.data.mem.ptr
                continue

        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


def pointer_adjust(func):
    """Decorator function for cupy/numpy agnostic cython.

    This decorator applies :func:`wrapper` to the arguments of every call before
    forwarding them to ``func``.

    If you use this decorator, you must convert input arrays to size_t data type in Cython and
    then properly cast the pointer as it enters the c++ function. See the
    Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    Args:
        func (Callable): Function taking pointer (``int``) arguments.

    Returns:
        Callable: Wrapper calling ``func(*targs, **tkwargs)``.
    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper
