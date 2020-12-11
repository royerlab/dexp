from dexp.processing.backends.backend import Backend
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel
from dexp.processing.registration.reg_trans_2d import register_translation_2d_dexp


def register_translation_maxproj_nd(image_a, image_b,
                                    register_translation_2d=register_translation_2d_dexp,
                                    gamma: float = 1,
                                    log_compression: bool = False,
                                    drop_worse: bool = True,
                                    internal_dtype=None,
                                    **kwargs):
    """
    Registers two nD (n=2 or 3) images using just a translation-only model.
    This method uses max projections along 2 or 3 axis and then performs phase correlation.

    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    register_translation_2d : 2d registration method to use
    gamma : gamma correction on max projections as a preprocessing before phase correlation.
    log_compression : Applies the function log1p to the images to compress high-intensities (usefull when very (too) bright structures are present in the images, such as beads)
    internal_dtype : internal dtype for computation


    Returns
    -------
    Translation-only registration model

    """

    xp = Backend.get_xp_module()

    if image_a.ndim != image_b.ndim:
        raise ValueError("Images must have the same number of dimensions")

    image_a = Backend.to_backend(image_a)
    image_b = Backend.to_backend(image_b)

    if image_a.ndim == 2:
        image_a = _preprocess_image(image_a, gamma=gamma, log_compression=log_compression, in_place=False, dtype=internal_dtype)
        image_b = _preprocess_image(image_b, gamma=gamma, log_compression=log_compression, in_place=False, dtype=internal_dtype)
        shifts, confidence = register_translation_2d(image_a, image_b, internal_dtype=internal_dtype, **kwargs).get_shift_and_confidence()

    elif image_a.ndim == 3:
        iap0 = _project_preprocess_image(image_a, axis=0, dtype=xp.float32, gamma=gamma, log_compression=log_compression)
        iap1 = _project_preprocess_image(image_a, axis=1, dtype=xp.float32, gamma=gamma, log_compression=log_compression)
        iap2 = _project_preprocess_image(image_a, axis=2, dtype=xp.float32, gamma=gamma, log_compression=log_compression)

        ibp0 = _project_preprocess_image(image_b, axis=0, dtype=xp.float32, gamma=gamma, log_compression=log_compression)
        ibp1 = _project_preprocess_image(image_b, axis=1, dtype=xp.float32, gamma=gamma, log_compression=log_compression)
        ibp2 = _project_preprocess_image(image_b, axis=2, dtype=xp.float32, gamma=gamma, log_compression=log_compression)

        shifts_p0, confidence_p0 = register_translation_2d(iap0, ibp0, internal_dtype=internal_dtype, **kwargs).get_shift_and_confidence()
        shifts_p1, confidence_p1 = register_translation_2d(iap1, ibp1, internal_dtype=internal_dtype, **kwargs).get_shift_and_confidence()
        shifts_p2, confidence_p2 = register_translation_2d(iap2, ibp2, internal_dtype=internal_dtype, **kwargs).get_shift_and_confidence()

        # print(shifts_p0)
        # print(shifts_p1)
        # print(shifts_p2)

        if drop_worse:
            worse_index = xp.argmin(xp.asarray([confidence_p0, confidence_p1, confidence_p2]))

            if worse_index == 0:
                shifts = xp.asarray([0.5 * (shifts_p1[0] + shifts_p2[0]), shifts_p2[1], shifts_p1[1]])
                confidence = (confidence_p1 * confidence_p2) ** 0.5
            elif worse_index == 1:
                shifts = xp.asarray([shifts_p2[0], 0.5 * (shifts_p0[0] + shifts_p2[1]), shifts_p0[1]])
                confidence = (confidence_p0 * confidence_p2) ** 0.5
            elif worse_index == 2:
                shifts = xp.asarray([shifts_p1[0], shifts_p0[0], 0.5 * (shifts_p0[1] + shifts_p1[1])])
                confidence = (confidence_p0 * confidence_p1) ** 0.5

        else:
            shifts_p0 = xp.asarray([0, shifts_p0[0], shifts_p0[1]])
            shifts_p1 = xp.asarray([shifts_p1[0], 0, shifts_p1[1]])
            shifts_p2 = xp.asarray([shifts_p2[0], shifts_p2[1], 0])
            shifts = (shifts_p0 + shifts_p1 + shifts_p2) / 2
            confidence = (confidence_p0 * confidence_p1 * confidence_p2) ** 0.33

        # if confidence>0.1:
        #     print(f"shift={shifts}, confidence={confidence}")

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
    else:
        raise ValueError(f'Unsupported number of dimensions ({image_a.ndim}) for registartion.')

    return TranslationRegistrationModel(shift_vector=shifts, confidence=confidence)


def _project_preprocess_image(image,
                              axis: int,
                              smoothing: float = 0,
                              quantile: int = None,
                              gamma: float = 1,
                              log_compression: bool = False,
                              dtype=None):
    image_projected = _project_image(image, axis=axis)
    image_projected_processed = _preprocess_image(image_projected,
                                                  smoothing=smoothing,
                                                  quantile=quantile,
                                                  gamma=gamma,
                                                  log_compression=log_compression,
                                                  dtype=dtype)

    return image_projected_processed


def _project_image(image, axis: int):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()
    image = Backend.to_backend(image)
    projection = xp.max(image, axis=axis) - xp.min(image, axis=axis)
    return projection


def _preprocess_image(image,
                      smoothing: float = 0,
                      log_compression: bool = True,
                      quantile: float = 0.01,
                      gamma: float = 1,
                      in_place: bool = True,
                      dtype=None):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    processed_image = Backend.to_backend(image, dtype=dtype, force_copy=not in_place)

    if smoothing > 0:
        processed_image = sp.ndimage.gaussian_filter(processed_image, sigma=smoothing)

    if log_compression:
        processed_image = xp.log1p(processed_image, out=processed_image)

    if quantile is None:
        min_value = processed_image.min()
        max_value = processed_image.max()
    else:
        min_value = xp.percentile(processed_image, q=100 * quantile)
        max_value = xp.percentile(processed_image, q=100 * (1 - quantile))

    alpha = max_value - min_value
    if alpha > 0:
        processed_image -= min_value
        processed_image /= alpha
        processed_image = xp.clip(processed_image, 0, 1, out=processed_image)

    if gamma != 1:
        processed_image **= gamma

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(processed_image), name='processed_image')

    return processed_image
