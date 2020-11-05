from typing import Union, Optional

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.filters.tenengrad import tenengrad
from dexp.processing.utils.blend import blend_arrays
from dexp.processing.utils.fit_shape import fit_shape


def fuse_tg_nd(backend: Backend,
               image_a,
               image_b,
               downscale: Optional[int] = 2,
               sharpness: Optional[float] = 24,
               tenenegrad_smoothing: Optional[int] = 4,
               blend_map_smoothing: Optional[int] = 10,
               bias_axis: Optional[int] = None,
               bias_exponent: Optional[float] = 3,
               bias_strength: Optional[float] = 2,
               clip: Optional[bool] = True):
    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    original_dtype = image_a.dtype

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)

    # downscale to speed up computation and reduce noise
    d_image_a = sp.ndimage.interpolation.zoom(image_a, zoom=1 / downscale, order=0)
    d_image_b = sp.ndimage.interpolation.zoom(image_b, zoom=1 / downscale, order=0)

    d_image_a = d_image_a.astype(xp.float32)
    d_image_b = d_image_b.astype(xp.float32)

    # Denoise further:
    d_image_a = sp.ndimage.gaussian_filter(d_image_a, sigma=1.5)
    d_image_b = sp.ndimage.gaussian_filter(d_image_b, sigma=1.5)

    # Compute Tenengrad filter:
    t_image_a = tenengrad(backend, d_image_a)
    t_image_b = tenengrad(backend, d_image_b)
    del d_image_a, d_image_b

    # Apply maximum filter:
    t_image_a = sp.ndimage.maximum_filter(t_image_a, size=tenenegrad_smoothing)
    t_image_b = sp.ndimage.maximum_filter(t_image_b, size=tenenegrad_smoothing)

    # Apply smoothing filter:
    t_image_a = sp.ndimage.uniform_filter(t_image_a, size=max(1, tenenegrad_smoothing))
    t_image_b = sp.ndimage.uniform_filter(t_image_b, size=max(1, tenenegrad_smoothing))

    # Normalise:
    t_min_value = min(xp.min(t_image_a), xp.min(t_image_b))
    t_max_value = max(xp.max(t_image_a), xp.max(t_image_b))
    t_image_a -= t_min_value
    t_image_b -= t_min_value
    t_image_a /= (t_max_value - t_min_value)
    t_image_b /= (t_max_value - t_min_value)

    # Add bias:
    if bias_axis is not None:
        length = t_image_a.shape[bias_axis]
        bias_vector = xp.linspace(-1, 1, num=length)
        bias_vector = xp.sign(bias_vector)*xp.absolute(bias_vector)**bias_exponent
        new_shape = tuple(s if i == bias_axis else 1 for i, s in enumerate(t_image_a.shape))
        bias_vector = xp.reshape(bias_vector, newshape=new_shape)
        t_image_a -= bias_strength * bias_vector
        t_image_b += bias_strength * bias_vector

    # compute the absolute difference and sign:
    diff = t_image_a - t_image_b
    del t_image_a, t_image_b, t_min_value, t_max_value
    abs_diff = xp.absolute(diff) ** (1 / sharpness)
    sgn_diff = xp.sign(diff)
    del diff

    # compute blending map:
    blend_map = 0.5 + 0.5 * sgn_diff * abs_diff
    del sgn_diff, abs_diff

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(d_image_a), name='image_a', contrast_limits=(0,600))
    #     viewer.add_image(_c(d_image_b), name='image_b', contrast_limits=(0,600))
    #     viewer.add_image(_c(t_image_a), name='t_image_a', contrast_limits=(0,1))
    #     viewer.add_image(_c(t_image_b), name='t_image_b', contrast_limits=(0,1))
    #     viewer.add_image(_c(blend_map), name='blend_map', contrast_limits=(0,1))

    # Upscale blending map back to original size:
    blend_map = sp.ndimage.zoom(blend_map, zoom=downscale, order=0)
    if type(backend) is CupyBackend:
        # Workarpound for lack of support for float16 in numpy:
        blend_map = blend_map.astype(xp.float16)

    # Padding to recover original image size:
    blend_map = fit_shape(backend, blend_map, shape=image_a.shape)

    # Smooth blend map to have less seams:
    blend_map = sp.ndimage.filters.uniform_filter(blend_map, size=blend_map_smoothing)

    # Fuse using blending map:
    image_fused = blend_arrays(backend, image_a, image_b, blend_map)

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value, out=image_fused)

    # Adjust type:
    image_fused = image_fused.astype(original_dtype)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image_a), name='image_a', contrast_limits=(0,600))
    #     viewer.add_image(_c(image_b), name='image_b', contrast_limits=(0,600))
    #     viewer.add_image(_c(blend_map), name='blend_map')
    #     viewer.add_image(_c(image_fused), name='image_fused', contrast_limits=(0,600))

    return image_fused
