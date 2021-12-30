from typing import Union

import numpy

try:
    import cupy

    xpArray = Union[cupy.ndarray, numpy.ndarray]
except ImportError:
    xpArray = numpy.ndarray
