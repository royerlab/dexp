import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.utils.element_wise_affine import element_wise_affine


def normalise(backend:Backend, image, low=0., high=1., out=None, internal_dtype=numpy.float16):
    xp = backend.get_xp_module()
    min_value = xp.min(image)
    max_value = xp.max(image)
    alpha = ((high-low) / (max_value - min_value)).astype(internal_dtype)
    beta = (low-alpha*min_value).astype(internal_dtype)
    image = element_wise_affine(backend, image, alpha, beta, out=out)

    def denorm_function(image):
        den_image, _ = normalise(backend, image, low=min_value, high=max_value, out=image if out is not None else None)
        return den_image

    return image, denorm_function