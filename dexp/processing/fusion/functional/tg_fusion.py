from dexp.processing.backends.backend import Backend
from dexp.processing.filters.tenengrad import tenengrad


def fuse_tg_nd(backend: Backend, image_a, image_b,
               downscale:int = 2, sharpness:float = 24, tenenegrad_smoothing=4, blend_map_smoothing=10,
               clip: bool = True):

    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    image_a = backend.to_backend(image_a)
    image_b = backend.to_backend(image_b)

    xp = backend.get_xp_module(image_a)
    sp = backend.get_sp_module(image_a)

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)

    # downscale to speed up computation and reduce noise
    t_image_a = sp.ndimage.interpolation.zoom(image_a, zoom=1/downscale)
    t_image_b = sp.ndimage.interpolation.zoom(image_b, zoom=1/downscale)

    # Compute Tenengrad filter:
    t_image_a = tenengrad(backend, t_image_a)
    t_image_b = tenengrad(backend, t_image_b)

    # Apply maximum filter:
    t_image_a = sp.ndimage.maximum_filter(t_image_a, size=tenenegrad_smoothing)
    t_image_b = sp.ndimage.maximum_filter(t_image_b, size=tenenegrad_smoothing)

    # Apply smoothing filter:
    t_image_a = sp.ndimage.uniform_filter(t_image_a, size=1+tenenegrad_smoothing//2)
    t_image_b = sp.ndimage.uniform_filter(t_image_b, size=1+tenenegrad_smoothing//2)

    # Normalise:
    t_min_value = min(xp.min(t_image_a), xp.min(t_image_b))
    t_max_value = min(xp.max(t_image_a), xp.max(t_image_b))
    t_image_a -= t_min_value
    t_image_b -= t_min_value
    t_image_a /= (t_max_value-t_min_value)
    t_image_b /= (t_max_value-t_min_value)

    # compute the absolute difference and sign:
    diff = t_image_a-t_image_b
    abs_diff = xp.absolute(diff) ** (1/sharpness)
    sgn_diff = xp.sign(diff)

    # compute blending map:
    blend_map = 0.5+0.5*sgn_diff*abs_diff

    # Upscale blending map back to original size:
    blend_map = sp.ndimage.zoom(blend_map, zoom=downscale)

    # Smooth blend map to have less seams:
    blend_map = sp.ndimage.uniform_filter(blend_map, size=blend_map_smoothing)

    # Fuse:
    image_fused = blend_map*image_a + (1-blend_map)*image_b

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #     viewer = Viewer()
    #     viewer.add_image(_c(image_a), name='image_a')
    #     viewer.add_image(_c(image_b), name='image_b')
    #     viewer.add_image(_c(t_image_a), name='t_image_a')
    #     viewer.add_image(_c(t_image_b), name='t_image_b')
    #     viewer.add_image(_c(blend_map), name='blend_map')
    #     viewer.add_image(_c(image_fused), name='image_fused')

    return image_fused







