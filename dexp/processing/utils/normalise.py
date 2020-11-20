from typing import Tuple

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.element_wise_affine import element_wise_affine


def normalise(backend: Backend,
              image,
              low: float = 0.,
              high: float = 1.,
              minmax: Tuple[float, float] = None,
              clip: bool = True,
              out=None,
              dtype=numpy.float16):
    xp = backend.get_xp_module()

    if type(backend) is NumpyBackend:
        dtype = numpy.float32

    image = backend.to_backend(image, dtype=dtype)

    if minmax is None:
        min_value = xp.min(image)
        max_value = xp.max(image)
    else:
        min_value, max_value = minmax

    alpha = ((high - low) / (max_value - min_value)).astype(dtype)
    beta = (low - alpha * min_value).astype(dtype)
    image = element_wise_affine(backend, image, alpha, beta, out=out)
    if clip:
        image = image.clip(low, high, out=image)

    def denorm_function(image):
        den_image, _ = normalise(backend, image, low=min_value, high=max_value, minmax=(low, high), clip=clip, out=image if out is not None else None, dtype=dtype)
        return den_image

    image = image.astype(dtype=dtype, copy=False)

    return image, denorm_function
