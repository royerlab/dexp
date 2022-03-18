from typing import Dict, Union

import numpy

try:
    import cupy

    xpArray = Union[cupy.ndarray, numpy.ndarray]
except ImportError:
    xpArray = numpy.ndarray

from dexp.utils.backends.cupy_backend import is_cupy_available


def dict_or(lhs: Dict, rhs: Dict) -> Dict:
    """
    Can be removed once we support only >= python 3.9
    https://peps.python.org/pep-0584/
    """
    lhs = lhs.copy()
    for k, v in rhs.items():
        if k not in lhs:
            lhs[k] = v

    return lhs
