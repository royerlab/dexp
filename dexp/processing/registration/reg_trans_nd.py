import math
from functools import reduce

import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.filters.sobel_filter import sobel_filter
from dexp.processing.registration.model.translation_registration_model import TranslationRegistrationModel


def register_translation_nd(backend: Backend,
                            image_a,
                            image_b,
                            max_range_ratio: float = 0.9,
                            fine_window_radius: int = 0,
                            decimate: int = 16,
                            quantile: float = 0.999,
                            sigma: float = 1.0,
                            edge_filter: bool = True,
                            internal_dtype=None) -> TranslationRegistrationModel:
    """
    Registers two nD images using just a translation-only model.
    This uses a full nD robust phase correlation based approach.


    Parameters
    ----------
    backend : backend for computation
    image_a : First image to register
    image_b : Second image to register
    max_range_ratio : backend for computation
    fine_window_radius : Window of which to refine translation estimate
    decimate : How much to decimate when computing floor level
    quantile : Quantile to use for robust min and max
    sigma : sigma for Gaussian smoothing of phase correlogram
    edge_filter : apply sobel edge filter to input images.
    internal_dtype : internal dtype for computation

    Returns
    -------
    Translation-only registration model

    """
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    if not image_a.dtype == image_b.dtype:
        raise ValueError("Arrays must have the same dtype")

    if internal_dtype is None:
        internal_dtype = image_a.dtype

    if type(backend) is NumpyBackend:
        internal_dtype = xp.float32

    image_a = backend.to_backend(image_a, dtype=internal_dtype)
    image_b = backend.to_backend(image_b, dtype=internal_dtype)

    if edge_filter:
        image_a = sobel_filter(backend,
                               image_a,
                               exponent=1,
                               normalise_input=False)
        image_b = sobel_filter(backend,
                               image_b,
                               exponent=1,
                               normalise_input=False)

    # We compute the phase correlation:
    raw_correlation = _phase_correlation(backend, image_a, image_b, internal_dtype)
    correlation = raw_correlation

    # max range is computed from max_range_ratio:
    max_ranges = tuple(int(0.5 * max_range_ratio * s) for s in correlation.shape)
    # print(f"max_ranges={max_ranges}")
    # We estimate the noise floor of the correlation:
    center = tuple(s // 2 for s in correlation.shape)
    empty_region = correlation[tuple(slice(0, c - r) for c, r in zip(center, max_ranges))]
    noise_floor_level = xp.percentile(empty_region.ravel()[::decimate].astype(numpy.float32), q=100 * quantile)
    # print(f"noise_floor_level={noise_floor_level}")

    # we use that floor to clip anything below:
    correlation = correlation.clip(noise_floor_level, math.inf) - noise_floor_level

    # We roll the array and crop it to restrict ourself to the search region:
    correlation = correlation[tuple(slice(c - r, c + r) for c, r in zip(center, max_ranges))]

    # denoise cropped correlation image:
    if sigma > 0:
        correlation = sp.ndimage.filters.gaussian_filter(correlation, sigma=sigma, mode='wrap')

    # We use the max as quickly computed proxy for the real center:
    max_correlation_flat_index = xp.argmax(correlation, axis=None)
    rough_shift = xp.unravel_index(max_correlation_flat_index, correlation.shape)
    max_correlation = correlation[rough_shift]

    # We compute the signed rough shift
    signed_rough_shift = xp.array(tuple(int(rs) - r for rs, r in zip(rough_shift, max_ranges)))
    signed_rough_shift = backend.to_numpy(signed_rough_shift)
    # print(f"signed_rough_shift= {signed_rough_shift}")

    # Compute confidence:
    masked_correlation = correlation.copy()
    masked_correlation[tuple(slice(rs - s // 8, rs + s // 8) for rs, s in zip(rough_shift, masked_correlation.shape))] = 0
    background_correlation_max = xp.max(masked_correlation)
    confidence = (max_correlation - background_correlation_max) / max_correlation
    # print(f"shift={signed_rough_shift}, confidence={confidence}")

    if fine_window_radius > 0:

        # We crop further to facilitate center-of-mass estimation:
        cropped_correlation = correlation[
            tuple(
                slice(max(0, int(s) - fine_window_radius), min(d, int(s) + fine_window_radius))
                for s, d in zip(rough_shift, correlation.shape)
            )
        ]
        cropped_correlation = backend.to_numpy(cropped_correlation)

        # We compute the center of mass:
        # We take the square to squash small values far from the maximum that are likely noisy...
        signed_com_shift = (
                xp.array(_center_of_mass(backend, cropped_correlation ** 2))
                - fine_window_radius
        )
        signed_com_shift = backend.to_numpy(signed_com_shift)
        # print(f"signed_com_shift= {signed_com_shift}")
        # The final shift is the sum of the rough sight plus the fine center of mass shift:
        shift = list(signed_rough_shift + signed_com_shift)
    else:
        cropped_correlation = None
        shift = list(signed_rough_shift)

    # # DO NOT DELETE, INSTRUMENTATION CODE FOR DEBUGGING
    # from napari import gui_qt, Viewer
    # with gui_qt():
    #     print(f"shift = {shift}")
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image_a), name='image_a')
    #     viewer.add_image(_c(image_b), name='image_b')
    #     viewer.add_image(_c(raw_correlation), name='raw_correlation', colormap='viridis')
    #     viewer.add_image(_c(correlation), name='correlation', colormap='viridis')
    #     if cropped_correlation is not None:
    #         viewer.add_image(_c(cropped_correlation), name='cropped_correlation', colormap='viridis')
    #     viewer.add_image(_c(masked_correlation), name='masked_correlation', colormap='viridis')
    #     viewer.grid_view(2,3,1)

    return TranslationRegistrationModel(shift_vector=shift, confidence=confidence)


def _center_of_mass(backend: Backend, image):
    image = backend.to_backend(image)

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    try:
        return sp.ndimage.center_of_mass(image)
    except AttributeError:
        # Workaround for the lack of implementation of center_of_mass in cupy
        # TODO: remove this code path once center_of_mass is implemented in cupy!

        normalizer = xp.sum(image)

        if abs(normalizer) > 0:
            grids = numpy.ogrid[[slice(0, i) for i in image.shape]]
            grids = list([backend.to_backend(grid) for grid in grids])

            results = list([xp.sum(image * grids[dir].astype(float)) / normalizer
                            for dir in range(image.ndim)])

            return tuple(float(f) for f in results)
        else:
            return tuple(s / 2 for s in image.shape)


def _normalised_projection(backend: Backend, image, axis, gamma=3):
    xp = backend.get_xp_module(image)
    projection = xp.max(image, axis=axis)
    min_value = xp.min(projection)
    max_value = xp.max(projection)
    range_value = (max_value - min_value)
    if abs(range_value) > 0:
        normalised_image = ((projection - min_value) / range_value) ** gamma
    else:
        normalised_image = 0
    return normalised_image


def _phase_correlation(backend: Backend,
                       image_a, image_b,
                       internal_dtype=numpy.float32,
                       epsilon: float = 1e-6,
                       window: float = 0.5):
    xp = backend.get_xp_module(image_a)

    if window > 0:
        window_axis = tuple(xp.hanning(s) ** window for s in image_a.shape)
        window = reduce(xp.multiply, xp.ix_(*window_axis))
        image_a *= window
        image_b *= window

    G_a = xp.fft.fftn(image_a).astype(numpy.complex64, copy=False)
    G_b = xp.fft.fftn(image_b).astype(numpy.complex64, copy=False)
    conj_b = xp.conj(G_b)
    R = G_a * conj_b
    R /= (xp.absolute(R) + epsilon)
    r = xp.fft.ifftn(R).real.astype(internal_dtype, copy=False)
    r = xp.fft.fftshift(r)
    return r
