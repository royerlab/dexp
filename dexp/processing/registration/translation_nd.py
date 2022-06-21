from functools import reduce

import numpy
from arbol import aprint

from dexp.processing.filters.sobel_filter import sobel_filter
from dexp.processing.registration.model.translation_registration_model import (
    TranslationRegistrationModel,
)
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def register_translation_nd(
    image_a: xpArray,
    image_b: xpArray,
    denoise_input_sigma: float = 1.5,
    gamma: float = 1,
    log_compression: bool = True,
    edge_filter: bool = True,
    max_range_ratio: float = 0.9,
    decimate: int = 16,
    quantile: float = 0.999,
    sigma: float = 1.5,
    force_numpy: bool = False,
    internal_dtype=None,
    _display_phase_correlation: bool = False,
) -> TranslationRegistrationModel:
    """
    Registers two nD images using just a translation-only model.
    This uses a full nD robust phase correlation based approach.

    Note: it is recommended to normalise the images to [0, 1]
        range before registering them, otherwise there might be precision and overflow issues.

    Parameters
    ----------
    image_a : First image to register
    image_b : Second image to register
    denoise_input_sigma : Uses a Gaussian filter to denoise input images.
    gamma : gamma correction on max projections as a preprocessing before phase correlation.
    log_compression : Applies the function log1p to the images to compress high-intensities
        (usefull when very (too) bright structures are present in the images, such as beads).
    edge_filter : apply sobel edge filter to input images.
    max_range_ratio : maximal range for correlation.
    decimate : How much to decimate when computing floor level
    quantile : Quantile to use for robust min and max
    sigma : sigma for Gaussian smoothing of phase correlogram
    force_numpy : Forces output model to be allocated with numpy arrays.
    internal_dtype : internal dtype for computation

    _display_phase_correlation : For debugging purposes the phase correlation can be displayed with napari

    Returns
    -------
    Translation-only registration model

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if not image_a.dtype == image_b.dtype:
        raise ValueError("Arrays must have the same dtype")

    if internal_dtype is None:
        internal_dtype = image_a.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    image_a = Backend.to_backend(image_a, dtype=internal_dtype)
    image_b = Backend.to_backend(image_b, dtype=internal_dtype)

    if denoise_input_sigma is not None and denoise_input_sigma > 0:
        image_a = sp.ndimage.gaussian_filter(image_a, sigma=denoise_input_sigma)
        image_b = sp.ndimage.gaussian_filter(image_b, sigma=denoise_input_sigma)

    if log_compression is not None and log_compression:
        image_a = xp.log1p(image_a)
        image_b = xp.log1p(image_b)

    if gamma is not None and gamma != 1:
        image_a **= gamma
        image_b **= gamma

    if edge_filter is not None and edge_filter:
        image_a = sobel_filter(image_a, exponent=1, normalise_input=False)
        image_b = sobel_filter(image_b, exponent=1, normalise_input=False)

    # Compute the phase correlation:
    raw_correlation = _phase_correlation(image_a, image_b, internal_dtype)
    correlation = raw_correlation

    # Max range is computed from max_range_ratio:
    max_ranges = tuple(int(0.5 * max_range_ratio * s) for s in correlation.shape)
    # print(f"max_ranges={max_ranges}")

    # We estimate the noise floor of the correlation:
    center = tuple(s // 2 for s in correlation.shape)
    empty_region = correlation[tuple(slice(0, c - r) for c, r in zip(center, max_ranges))]
    noise_floor_level = xp.quantile(empty_region.ravel()[::decimate].astype(numpy.float32), q=quantile)
    if xp.isnan(noise_floor_level):
        noise_floor_level = xp.mean(empty_region.ravel()[::decimate])
    # print(f"noise_floor_level={noise_floor_level}")

    # Roll the array and crop it to restrict ourself to the search region:
    correlation = correlation[
        tuple(slice(max(c - r, 0), min(c + r, s)) for c, r, s in zip(center, max_ranges, correlation.shape))
    ]

    # Use that floor to clip anything below:
    correlation = xp.maximum(correlation, noise_floor_level, out=correlation)
    correlation -= noise_floor_level

    # Denoise cropped correlation image:
    if sigma > 0:
        correlation = sp.ndimage.gaussian_filter(correlation, sigma=sigma, mode="wrap")

    # Use the max as quickly computed proxy for the real center:
    max_correlation_flat_index = xp.argmax(correlation, axis=None)
    rough_shift = xp.unravel_index(max_correlation_flat_index, correlation.shape)
    max_correlation = correlation[rough_shift]

    # Compute the signed shift vector:
    shift_vector = xp.array(tuple(int(rs) - r for rs, r in zip(rough_shift, max_ranges)))

    # Compute confidence:
    masked_correlation = correlation.copy()
    mask_size = tuple(max(8, int(s**0.9) // 8) for s in masked_correlation.shape)
    masked_correlation[tuple(slice(rs - s, rs + s) for rs, s in zip(rough_shift, mask_size))] = 0
    background_correlation_max = xp.max(masked_correlation)
    epsilon = 1e-6
    confidence = (max_correlation - background_correlation_max) / (epsilon + max_correlation)

    if _display_phase_correlation:
        # DO NOT DELETE, INSTRUMENTATION CODE FOR DEBUGGING
        from napari import Viewer, gui_qt

        with gui_qt():
            aprint(f"shift = {shift_vector}, confidence = {confidence} ")

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_a), name="image_a")
            viewer.add_image(_c(image_b), name="image_b")
            viewer.add_image(_c(raw_correlation), name="raw_correlation", colormap="viridis")
            viewer.add_image(_c(correlation), name="correlation", colormap="viridis")
            viewer.add_image(
                _c(masked_correlation), name="masked_correlation", colormap="bop orange", blending="additive"
            )
            viewer.grid.enabled = True
            viewer.grid.shape = (2, 3)

    return TranslationRegistrationModel(shift_vector=shift_vector, confidence=confidence, force_numpy=force_numpy)


def _center_of_mass(image):
    image = Backend.to_backend(image)

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    try:
        return sp.ndimage.center_of_mass(image)
    except AttributeError:
        # Workaround for the lack of implementation of center_of_mass in cupy
        # TODO: remove this code path once center_of_mass is implemented in cupy!

        normalizer = xp.sum(image)

        if abs(normalizer) > 0:
            grids = numpy.ogrid[[slice(0, i) for i in image.shape]]
            grids = list(Backend.to_backend(grid) for grid in grids)

            results = list(xp.sum(image * grids[dir].astype(float)) / normalizer for dir in range(image.ndim))

            return tuple(float(f) for f in results)
        else:
            return tuple(s / 2 for s in image.shape)


def _phase_correlation(image_a, image_b, internal_dtype=numpy.float32, epsilon: float = 1e-6, window: float = 0.5):
    xp = Backend.get_xp_module(image_a)

    if window > 0:
        window_axis = tuple(xp.hanning(s) ** window for s in image_a.shape)
        window = reduce(xp.multiply, xp.ix_(*window_axis))
        image_a *= window
        image_b *= window

    G_a = xp.fft.fftn(image_a).astype(numpy.complex64, copy=False)
    G_b = xp.fft.fftn(image_b).astype(numpy.complex64, copy=False)
    conj_b = xp.conj(G_b)
    R = G_a * conj_b
    R /= xp.absolute(R) + epsilon
    r = xp.fft.ifftn(R).real.astype(internal_dtype, copy=False)
    r = xp.fft.fftshift(r)
    return r
