import numpy
from typing import Union

try:
    import cupy
    xpArray = Union[cupy.ndarray, numpy.ndarray]
except ImportError:
    xpArray = numpy.ndarray
