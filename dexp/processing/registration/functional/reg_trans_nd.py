import math

import numpy
from dexp.processing.backends.backend import Backend


def register_translation_nd(backend: Backend, image_a, image_b, max_range_ratio=0.5, fine_window_radius=4, decimate=16, percentile=99.9, dtype=numpy.float32, log=False):

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module(image_a)
    sp = backend.get_sp_module(image_a)

    # We compute the phase correlation:
    raw_correlation = _phase_correlation(backend, image_a, image_b, dtype)
    correlation = raw_correlation

    # max range is computed from max_range_ratio:
    max_range = 8 + int(max_range_ratio*numpy.min(correlation.shape))

    # We estimate the noise floor of the correlation:
    max_ranges = tuple(max(0, min(max_range, s-2*max_range)) for s in correlation.shape)
    if log:
        print(f"max_ranges={max_ranges}")
    empty_region = correlation[tuple(slice(r, s-r) for r,s in zip(max_ranges, correlation.shape))].copy()
    noise_floor_level = xp.percentile(empty_region.ravel()[::decimate].astype(numpy.float32), q=percentile)
    if log:
        print(f"noise_floor_level={noise_floor_level}")

    # we use that floor to clip anything below:
    correlation = correlation.clip(noise_floor_level, math.inf) - noise_floor_level

    # We roll the array and crop it to restrict ourself to the search region:
    correlation = xp.roll(correlation, shift=max_range, axis=tuple(range(image_a.ndim)))
    correlation = correlation[(slice(0, 2 * max_range),) * image_a.ndim]

    # denoise cropped corelation image:
    #correlation = gaussian_filter(correlation, sigma=sigma, mode='wrap')

    # We use the max as quickly computed proxy for the real center:
    rough_shift = xp.unravel_index(
        xp.argmax(correlation, axis=None), correlation.shape
    )

    if log:
        print(f"rough_shift= {rough_shift}")


    # We crop further to facilitate center-of-mass estimation:
    cropped_correlation = correlation[
        tuple(
            slice(max(0, int(s) - fine_window_radius), min(d, int(s) + fine_window_radius))
            for s, d in zip(rough_shift, correlation.shape)
        )
    ]
    if log:
        print(f"cropped_correlation.shape = {cropped_correlation.shape}")

    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(self._cn(a), name='a')
    #     viewer.add_image(self._cn(b), name='b')
    #     viewer.add_image(self._cn(raw_correlation), name='raw_correlation')
    #     viewer.add_image(self._cn(correlation), name='correlation')
    #     viewer.add_image(self._cn(cropped_correlation), name='cropped_correlation')

    # We compute the signed rough shift
    signed_rough_shift = xp.array(rough_shift) - max_range
    if log:
        print(f"signed_rough_shift= {signed_rough_shift}")

    signed_rough_shift = backend.to_numpy(signed_rough_shift)
    cropped_correlation = backend.to_numpy(cropped_correlation)

    # We compute the center of mass:
    # We take the square to squash small values far from the maximum that are likely noisy...
    signed_com_shift = (
            numpy.array(sp.ndimage.center_of_mass(cropped_correlation ** 2))
            - fine_window_radius
    )
    if log:
        print(f"signed_com_shift= {signed_com_shift}")

    # The final shift is the sum of the rough sight plus the fine center of mass shift:
    shift = list(signed_rough_shift + signed_com_shift)

    if log:
        print(f"shift = {shift}")

    return shift, 0

def _normalised_projection(backend:Backend, image, axis, gamma=3):
    xp = backend.get_xp_module(image)
    projection = xp.max(image, axis=axis)
    min_value = xp.min(projection)
    max_value = xp.max(projection)
    normalised_image = ((projection-min_value) / (max_value - min_value)) ** gamma
    return normalised_image

def _phase_correlation(backend:Backend, image_a, image_b, dtype=numpy.float32):
    xp = backend.get_xp_module(image_a)
    G_a = xp.fft.fftn(image_a).astype(numpy.complex64, copy=False)
    G_b = xp.fft.fftn(image_b).astype(numpy.complex64, copy=False)
    conj_b = xp.conj(G_b)
    R = G_a * conj_b
    R /= xp.absolute(R)
    r = xp.fft.ifftn(R).real.astype(dtype, copy=False)
    return r
















