import numpy
from numpy.linalg import norm

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.functional.reg_trans_2d import register_translation_2d_skimage
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel


def register_translation_maxproj_nd(backend: Backend, image_a, image_b, register_translation_2d=register_translation_2d_skimage, gamma=4):

    if image_a.ndim != image_b.ndim:
        raise ValueError("Images must have the same number of dimensions")

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    if image_a.ndim == 2:
        shifts, error = register_translation_2d(backend, image_a, image_b)

    elif image_a.ndim == 3:
        iap0 = _normalised_projection(backend, image_a, axis=0, gamma=gamma)
        iap1 = _normalised_projection(backend, image_a, axis=1, gamma=gamma)
        iap2 = _normalised_projection(backend, image_a, axis=2, gamma=gamma)

        ibp0 = _normalised_projection(backend, image_b, axis=0, gamma=gamma)
        ibp1 = _normalised_projection(backend, image_b, axis=1, gamma=gamma)
        ibp2 = _normalised_projection(backend, image_b, axis=2, gamma=gamma)

        shifts_p0, error_p0 = register_translation_2d(backend, iap0, ibp0).get_shift_and_error()
        shifts_p1, error_p1 = register_translation_2d(backend, iap1, ibp1).get_shift_and_error()
        shifts_p2, error_p2 = register_translation_2d(backend, iap2, ibp2).get_shift_and_error()

        shifts_p0 = numpy.asarray([0, shifts_p0[0], shifts_p0[1]])
        shifts_p1 = numpy.asarray([shifts_p1[0], 0, shifts_p1[1]])
        shifts_p2 = numpy.asarray([shifts_p2[0], shifts_p2[1], 0])

        shifts = (shifts_p0+shifts_p1+shifts_p2)/2
        error = norm([error_p0, error_p1, error_p2])

    return TranslationRegistrationModel(shift_vector=shifts, error=error)

def _normalised_projection(backend:Backend, image, axis, gamma=3):
    image = backend.to_backend(image)
    xp = backend.get_xp_module(image)
    projection = xp.max(image, axis=axis)
    min_value = xp.min(projection)
    max_value = xp.max(projection)
    normalised_image = ((projection-min_value) / (max_value - min_value)) ** gamma
    return normalised_image














