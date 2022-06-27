from typing import Dict, List, Union

import numpy

try:
    import cupy

    xpArray = Union[cupy.ndarray, numpy.ndarray]
except ImportError:
    xpArray = numpy.ndarray


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


def overwrite2mode(overwrite: bool) -> str:
    """Returns writing mode from overwrite flag"""
    return "w" if overwrite else "w-"


def compress_dictionary_lists_length(d: Dict, max_length: int) -> Dict:
    """Reduces the length of lists in an dictionary and convert to string. Useful for printing"""
    out = {}
    for k, v in d.items():
        if isinstance(v, Dict):
            out[k] = compress_dictionary_lists_length(v, max_length)
        elif isinstance(v, List) and len(v) > max_length:
            out[k] = str(v[:max_length])[:-1] + ", ...]"
        else:
            out[k] = v

    return out
