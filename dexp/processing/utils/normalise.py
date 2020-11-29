from typing import Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.element_wise_affine import element_wise_affine


def normalise_functions(backend: Backend,
                        image,
                        low: float = 0.,
                        high: float = 1.,
                        minmax: Tuple[float, float] = None,
                        quantile: float = 0,
                        clip: bool = True,
                        do_normalise: bool = True,
                        in_place: bool = True,
                        dtype=None):
    xp = backend.get_xp_module()

    if dtype is None:
        if image.dtype.name == 'uint8' or \
                image.dtype.name == 'uint16' or \
                image.dtype.name == 'int8' or \
                image.dtype.name == 'uint16' or \
                image.dtype.name == 'float16':
            dtype = xp.float16
        elif image.dtype.name == 'uint32' or \
                image.dtype.name == 'int32' or \
                image.dtype.name == 'float32':
            dtype = xp.float32
        elif image.dtype.name == 'uint64' or \
                image.dtype.name == 'int64' or \
                image.dtype.name == 'float64':
            dtype = xp.float32

    if type(backend) is NumpyBackend:
        dtype = numpy.float32

    original_dtype = image.dtype

    if minmax is None:
        if quantile > 0:
            min_value = xp.percentile(image, q=100 * quantile)
            max_value = xp.percentile(image, q=100 * (1 - quantile))
        else:
            min_value = xp.min(image)
            max_value = xp.max(image)
    else:
        min_value, max_value = minmax

    # normalise:
    norm_alpha = ((high - low) / (max_value - min_value)).astype(dtype)
    norm_beta = (low - norm_alpha * min_value).astype(dtype)

    def _norm_function(_image):
        if not do_normalise:
            return _image
        _image = backend.to_backend(_image, dtype=dtype)
        _image = element_wise_affine(backend, _image, norm_alpha, norm_beta, out=_image if in_place else None)
        if clip:
            _image = _image.clip(low, high, out=_image)
        _image = _image.astype(dtype=dtype, copy=False)
        return _image

    # denormalise:
    denorm_alpha = ((max_value - min_value) / (high - low)).astype(dtype)
    denorm_beta = (min_value - denorm_alpha * low).astype(dtype)

    def _denorm_function(_image):
        if not do_normalise:
            return _image
        _image = backend.to_backend(_image, dtype=dtype)
        _image = element_wise_affine(backend, _image, denorm_alpha, denorm_beta, out=_image if in_place else None)
        if clip:
            _image = _image.clip(min_value, max_value, out=_image)
        _image = _image.astype(dtype=original_dtype, copy=False)
        return _image

    return _norm_function, _denorm_function
