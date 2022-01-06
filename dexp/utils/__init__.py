from typing import Union

import numpy

try:
    import cupy

    xpArray = Union[cupy.ndarray, numpy.ndarray]
except ImportError:
    xpArray = numpy.ndarray

from dexp.utils.backends.cupy_backend import is_cupy_available
