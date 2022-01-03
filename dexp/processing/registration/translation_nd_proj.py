from typing import Callable

from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.processing.registration.translation_2d import register_translation_2d_dexp
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def register_translation_proj_nd(
    image_a: xpArray,
    image_b: xpArray,
    register_translation_2d: Callable = register_translation_2d_dexp,
    drop_worse: bool = True,
    force_numpy: bool = False,
    internal_dtype=None,
    **kwargs,
):
    """
    Registers two nD (n=2 or 3) images using just a translation-only model.
    This method uses projections along 2 or 3 axis and then performs phase correlation.

    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    register_translation_2d : 2d registration method to use
    drop_worse: drops the worst 2D registrations before combining the projection
        registration vectors to a full nD registration vector.
    force_numpy : Forces output model to be allocated with numpy arrays.
    internal_dtype : Internal dtype for computation


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
        image_a = _preprocess_image(image_a, in_place=False, dtype=internal_dtype)
        image_b = _preprocess_image(image_b, in_place=False, dtype=internal_dtype)

        shifts, confidence = register_translation_2d(
            image_a, image_b, force_numpy=force_numpy, internal_dtype=internal_dtype, **kwargs
        ).get_shift_and_confidence()

    elif image_a.ndim == 3:
        iap0 = _project_preprocess_image(image_a, axis=0, dtype=xp.float32)
        iap1 = _project_preprocess_image(image_a, axis=1, dtype=xp.float32)
        iap2 = _project_preprocess_image(image_a, axis=2, dtype=xp.float32)

        ibp0 = _project_preprocess_image(image_b, axis=0, dtype=xp.float32)
        ibp1 = _project_preprocess_image(image_b, axis=1, dtype=xp.float32)
        ibp2 = _project_preprocess_image(image_b, axis=2, dtype=xp.float32)

        shifts_p0, confidence_p0 = register_translation_2d(
            iap0, ibp0, force_numpy=force_numpy, internal_dtype=internal_dtype, **kwargs
        ).get_shift_and_confidence()

        shifts_p1, confidence_p1 = register_translation_2d(
            iap1, ibp1, force_numpy=force_numpy, internal_dtype=internal_dtype, **kwargs
        ).get_shift_and_confidence()

        shifts_p2, confidence_p2 = register_translation_2d(
            iap2, ibp2, force_numpy=force_numpy, internal_dtype=internal_dtype, **kwargs
        ).get_shift_and_confidence()

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
        raise ValueError(f"Unsupported number of dimensions ({image_a.ndim}) for registration.")

    model = TranslationRegistrationModel(shift_vector=shifts, confidence=confidence, force_numpy=force_numpy)

    return model


def _project_preprocess_image(
    image, axis: int, smoothing: float = 0, quantile: int = None, gamma: float = 1, dtype=None
):
    image_projected = _project_image(image, axis=axis)
    image_projected_processed = _preprocess_image(image_projected, quantile=quantile, dtype=dtype)

    return image_projected_processed


def _project_image(image, axis: int):
    xp = Backend.get_xp_module()
    image = Backend.to_backend(image)
    projection = xp.max(image, axis=axis) - xp.min(image, axis=axis)
    return projection


def _preprocess_image(image, quantile: float = 0.01, in_place: bool = True, dtype=None):
    xp = Backend.get_xp_module()

    processed_image = Backend.to_backend(image, dtype=dtype, force_copy=not in_place)

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

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image), name='image')
    #     viewer.add_image(_c(processed_image), name='processed_image')

    return processed_image
