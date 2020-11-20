import numpy
from numpy.linalg import norm

from dexp.processing.backends.backend import Backend
from dexp.processing.filters.sobel import sobel_magnitude_filter
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.reg_trans_2d import register_translation_2d_dexp


def register_translation_maxproj_nd(backend: Backend,
                                    image_a, image_b,
                                    register_translation_2d=register_translation_2d_dexp,
                                    gamma=2,
                                    internal_dtype=None):
    """
    Registers two nD (n=2 or 3) images using just a translation-only model.
    This method uses max projections along 2 or 3 axis and then performs phase correlation.

    Parameters
    ----------
    backend : backend for computation
    image_a : First image to register
    image_b : Second image to register
    register_translation_2d : 2d registration method to use
    gamma : gamma correstion on max projections as a preprocessing before phase correlation.
    internal_dtype : internal dtype for computation


    Returns
    -------
    Translation-only registration model

    """

    if image_a.ndim != image_b.ndim:
        raise ValueError("Images must have the same number of dimensions")

    xp = backend.get_xp_module()

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    if image_a.ndim == 2:
        image_a = _preprocess_image(backend, image_a, gamma=gamma, dtype=xp.float32)
        image_b = _preprocess_image(backend, image_b, gamma=gamma, dtype=xp.float32)
        shifts, error = register_translation_2d(backend, image_a, image_b).get_shift_and_error()

    elif image_a.ndim == 3:
        iap0 = _project_preprocess_image(backend, image_a, axis=0, dtype=xp.float32, gamma=gamma)
        iap1 = _project_preprocess_image(backend, image_a, axis=1, dtype=xp.float32, gamma=gamma)
        iap2 = _project_preprocess_image(backend, image_a, axis=2, dtype=xp.float32, gamma=gamma)

        ibp0 = _project_preprocess_image(backend, image_b, axis=0, dtype=xp.float32, gamma=gamma)
        ibp1 = _project_preprocess_image(backend, image_b, axis=1, dtype=xp.float32, gamma=gamma)
        ibp2 = _project_preprocess_image(backend, image_b, axis=2, dtype=xp.float32, gamma=gamma)

        shifts_p0, error_p0 = register_translation_2d(backend, iap0, ibp0, internal_dtype=internal_dtype).get_shift_and_error()
        shifts_p1, error_p1 = register_translation_2d(backend, iap1, ibp1, internal_dtype=internal_dtype).get_shift_and_error()
        shifts_p2, error_p2 = register_translation_2d(backend, iap2, ibp2, internal_dtype=internal_dtype).get_shift_and_error()

        shifts_p0 = numpy.asarray([0, shifts_p0[0], shifts_p0[1]])
        shifts_p1 = numpy.asarray([shifts_p1[0], 0, shifts_p1[1]])
        shifts_p2 = numpy.asarray([shifts_p2[0], shifts_p2[1], 0])

        # print(shifts_p0)
        # print(shifts_p1)
        # print(shifts_p2)

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


def _project_preprocess_image(backend: Backend,
                              image,
                              axis: int,
                              smoothing: float = 0.5,
                              percentile: int = 1,
                              edge_filter: bool = True,
                              gamma: float = 3,
                              dtype=None):
    image_projected = _project_image(backend, image, axis=axis)
    image_projected_processed = _preprocess_image(backend,
                              image_projected,
                              smoothing=smoothing,
                              percentile=percentile,
                              edge_filter=edge_filter,
                              gamma=gamma,
                              dtype=dtype)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(image_projected), name='image_projected')
    #     viewer.add_image(_c(image_projected_processed), name='image_projected_processed')

    return image_projected_processed


def _project_image(backend: Backend, image, axis: int):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()
    image = backend.to_backend(image)
    projection = xp.max(image, axis=axis)
    return projection

def _preprocess_image(backend: Backend,
                      image,
                      smoothing: float = 0.5,
                      percentile: int = 1,
                      edge_filter: bool = True,
                      gamma: float = 3,
                      dtype=None):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()
    processed_image = backend.to_backend(image, dtype=dtype)

    if smoothing > 0:
        processed_image = sp.ndimage.gaussian_filter(processed_image, sigma=smoothing)

    min_value = xp.percentile(processed_image, q=percentile)
    max_value = xp.percentile(processed_image, q=100 - percentile)
    processed_image -= min_value
    processed_image *= 1 / (max_value - min_value)
    processed_image = xp.clip(processed_image, 0, 1, out=processed_image)

    if edge_filter:
        processed_image = sobel_magnitude_filter(backend, processed_image, exponent=1, in_place=True, normalise_input=False)

    processed_image **= gamma

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(processed_image), name='processed_image')

    return processed_image


