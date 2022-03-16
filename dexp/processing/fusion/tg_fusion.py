from typing import Optional

from dexp.processing.filters.sobel_filter import sobel_filter
from dexp.processing.utils.blend_images import blend_images
from dexp.processing.utils.element_wise_affine import element_wise_affine
from dexp.processing.utils.fit_shape import fit_to_shape
from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def fuse_tg_nd(
    image_a: xpArray,
    image_b: xpArray,
    downscale: Optional[int] = 2,
    sharpness: Optional[float] = 24,
    tenengrad_smoothing: Optional[int] = 4,
    blend_map_smoothing: Optional[int] = 10,
    bias_axis: Optional[int] = None,
    bias_exponent: Optional[float] = 3,
    bias_strength: Optional[float] = 2,
    clip: Optional[bool] = True,
    internal_dtype=None,
    _display_blend_map: bool = False,
):
    """
    Fuses two images by picking regions from one or the other image based on the local image quality
    measured by using the magnitude of the Sobel gradient -- similarly as in the Tenengrad focus metric.
    A smooth blend map is computed that blends the two images based on local image quality. A bias can be
    introduced to favor one side of an axis versus another.


    Parameters
    ----------
    image_a : First image to fuse
    image_b : Second image to fuse
    downscale : How much to downscale the two images in order to compute the blend map.
    A value of 2 is good to achieve both coarse denoising and reduce compute load.
    sharpness : How 'sharp' should be the choice between the two images.
    A large value makes sure that most of the time the voxel values of one or the other image
    are picked with very little mixing even if local image quality between .
    tenengrad_smoothing : How much to smooth the tenengrad map
    blend_map_smoothing : How much to smooth the blend map
    bias_axis : Along which axis should a bias be introduced in the blend map. None for no bias.
    bias_exponent : Exponent for the bias
    bias_strength : Bias strength -- zero means no bias
    clip : clip output to input images min and max values.
    internal_dtype : dtype for internal computation

    _display_blend_map : For debugging purposes, we can display the images to fuse, the blend map and result.

    Returns
    -------

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if not image_a.shape == image_b.shape:
        raise ValueError("Arrays must have the same shape")

    if not image_a.dtype == image_b.dtype:
        raise ValueError("Arrays must have the same dtype")

    if internal_dtype is None:
        internal_dtype = image_a.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    original_dtype = image_a.dtype

    image_a = Backend.to_backend(image_a, dtype=internal_dtype)
    image_b = Backend.to_backend(image_b, dtype=internal_dtype)

    min_a, max_a = xp.min(image_a), xp.max(image_a)
    min_b, max_b = xp.min(image_b), xp.max(image_b)
    min_value = min(min_a, min_b)
    max_value = min(max_a, max_b)
    del min_a, max_a, min_b, max_b

    # downscale to speed up computation and reduce noise
    d_image_a = sp.ndimage.zoom(image_a, zoom=1 / downscale, order=0)
    d_image_b = sp.ndimage.zoom(image_b, zoom=1 / downscale, order=0)

    # Denoise further:
    d_image_a = sp.ndimage.gaussian_filter(d_image_a, sigma=1.5)
    d_image_b = sp.ndimage.gaussian_filter(d_image_b, sigma=1.5)

    # Compute Tenengrad filter:
    t_image_a = sobel_filter(d_image_a, exponent=1, normalise_input=False, in_place_normalisation=True)
    t_image_b = sobel_filter(d_image_b, exponent=1, normalise_input=False, in_place_normalisation=True)
    del d_image_a, d_image_b

    # Apply maximum filter:
    t_image_a = sp.ndimage.maximum_filter(t_image_a, size=tenengrad_smoothing)
    t_image_b = sp.ndimage.maximum_filter(t_image_b, size=tenengrad_smoothing)

    # Apply smoothing filter:
    t_image_a = sp.ndimage.uniform_filter(t_image_a, size=max(1, tenengrad_smoothing))
    t_image_b = sp.ndimage.uniform_filter(t_image_b, size=max(1, tenengrad_smoothing))

    # Normalise:

    t_min_value = min(xp.min(t_image_a), xp.min(t_image_b))
    t_max_value = max(xp.max(t_image_a), xp.max(t_image_b))
    alpha = (1 / (t_max_value - t_min_value)).astype(internal_dtype)
    beta = (-t_min_value / (t_max_value - t_min_value)).astype(internal_dtype)
    t_image_a = element_wise_affine(t_image_a, alpha, beta, out=t_image_a)
    t_image_b = element_wise_affine(t_image_b, alpha, beta, out=t_image_b)
    del t_min_value, t_max_value

    # Add bias:
    if bias_axis is not None and bias_strength != 0:
        length = t_image_a.shape[bias_axis]
        bias_vector = xp.linspace(-1, 1, num=length)
        bias_vector = xp.sign(bias_vector) * (xp.absolute(bias_vector) ** bias_exponent)
        new_shape = tuple(s if i == bias_axis else 1 for i, s in enumerate(t_image_a.shape))
        bias_vector = xp.reshape(bias_vector, newshape=new_shape)
        t_image_a -= bias_strength * bias_vector
        t_image_b += bias_strength * bias_vector
        del bias_vector

    # compute the absolute difference and sign:
    diff = t_image_a
    diff -= t_image_b
    del t_image_b

    sgn_diff = xp.sign(diff)
    abs_diff = xp.absolute(diff, out=diff)
    abs_diff **= 1 / sharpness
    del diff

    # compute blending map:
    blend_map = abs_diff
    blend_map *= sgn_diff
    blend_map = element_wise_affine(blend_map, 0.5, 0.5, out=blend_map)
    del sgn_diff

    # Upscale blending map back to original size:
    blend_map = sp.ndimage.zoom(blend_map, zoom=downscale, order=1)

    # Padding to recover original image size:
    blend_map = fit_to_shape(blend_map, shape=image_a.shape)

    # Smooth blend map to have less seams:
    blend_map = sp.ndimage.uniform_filter(blend_map, size=blend_map_smoothing)

    # Fuse using blending map:
    image_fused = blend_images(image_a, image_b, blend_map)
    if not _display_blend_map:
        del image_a, image_b, blend_map

    if clip:
        image_fused = xp.clip(image_fused, min_value, max_value, out=image_fused)

    # Adjust type:
    image_fused = image_fused.astype(original_dtype, copy=False)

    if _display_blend_map:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image_a), name="image_a", contrast_limits=(0, 600))
            viewer.add_image(_c(image_b), name="image_b", contrast_limits=(0, 600))
            viewer.add_image(_c(blend_map), name="blend_map")
            viewer.add_image(_c(image_fused), name="image_fused", contrast_limits=(0, 600))

    return image_fused
