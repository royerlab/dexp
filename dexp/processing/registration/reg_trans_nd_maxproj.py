import numpy
from numpy.linalg import norm

from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.reg_trans_2d import register_translation_2d_dexp


def register_translation_maxproj_nd(backend: Backend, image_a, image_b, register_translation_2d=register_translation_2d_dexp, gamma=2):
    if image_a.ndim != image_b.ndim:
        raise ValueError("Images must have the same number of dimensions")

    xp = backend.get_xp_module()

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    if image_a.ndim == 2:
        shifts, error = register_translation_2d(backend, image_a, image_b)

    elif image_a.ndim == 3:
        iap0 = _normalised_projection(backend, image_a, axis=0, dtype=xp.float32, gamma=gamma)
        iap1 = _normalised_projection(backend, image_a, axis=1, dtype=xp.float32, gamma=gamma)
        iap2 = _normalised_projection(backend, image_a, axis=2, dtype=xp.float32, gamma=gamma)

        ibp0 = _normalised_projection(backend, image_b, axis=0, dtype=xp.float32, gamma=gamma)
        ibp1 = _normalised_projection(backend, image_b, axis=1, dtype=xp.float32, gamma=gamma)
        ibp2 = _normalised_projection(backend, image_b, axis=2, dtype=xp.float32, gamma=gamma)

        shifts_p0, error_p0 = register_translation_2d(backend, iap0, ibp0).get_shift_and_error()
        shifts_p1, error_p1 = register_translation_2d(backend, iap1, ibp1).get_shift_and_error()
        shifts_p2, error_p2 = register_translation_2d(backend, iap2, ibp2).get_shift_and_error()

        shifts_p0 = numpy.asarray([0, shifts_p0[0], shifts_p0[1]])
        shifts_p1 = numpy.asarray([shifts_p1[0], 0, shifts_p1[1]])
        shifts_p2 = numpy.asarray([shifts_p2[0], shifts_p2[1], 0])

        print(shifts_p0)
        print(shifts_p1)
        print(shifts_p2)

        shifts = (shifts_p0 + shifts_p1 + shifts_p2) / 2
        error = norm([error_p0, error_p1, error_p2])

        # from napari import Viewer, gui_qt
        # with gui_qt():
        #     def _c(array):
        #         return backend.to_numpy(array)
        #
        #     viewer = Viewer()
        #     viewer.add_image(_c(iap0), name='iap0')
        #     viewer.add_image(_c(ibp0), name='ibp0')
        #     viewer.add_image(_c(iap1), name='iap1')
        #     viewer.add_image(_c(ibp1), name='ibp1')
        #     viewer.add_image(_c(iap2), name='iap2')
        #     viewer.add_image(_c(ibp2), name='ibp2')

    return TranslationRegistrationModel(shift_vector=shifts, error=error)


def _normalised_projection(backend: Backend, image, axis, dtype=None, gamma=3, quantile=1):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()
    image = backend.to_backend(image)
    projection = xp.max(image, axis=axis)
    projection = projection.astype(dtype, copy=False)
    smoothed_projection = projection.copy()
    # smoothed_projection = sp.ndimage.gaussian_filter(smoothed_projection, sigma=1)
    min_value = xp.percentile(smoothed_projection, q=quantile)
    max_value = xp.percentile(smoothed_projection, q=100 - quantile)
    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(smoothed_projection), name='smoothed_projection')
    #     viewer.add_image(_c(projection), name='projection')

    projection -= min_value
    projection *= 1 / (max_value - min_value)
    projection = xp.clip(projection, 0, 1, out=projection)
    projection **= gamma
    return projection
