from typing import Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.border import add_border


def resize_color_image(image,
                       new_size: Tuple[int, int],
                       mode: str = 'canvas',
                       placement: str = 'center',
                       pad_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
                       preserve_aspect_ratio: bool = True,
                       rgba_value_max: float = 255):
    """
    Resizes an image to a given new size.
    There are two resizing modes: 'canvas' resizes the canvas around the ianmge without touching the original pixels, paddding or cropping is done as needed.
    In 'image' mode, the actual image is resized. Setting preserve_aspect_ratio to True preserves the aspect ratio,
    and a padding color is used where needed.



    Parameters
    ----------
    image: image to resize.
    new_size: New image size as a tuple (nw, nh) of integers.
    mode: Two modes are available: 'image' mode resizes the image itself, possibly preserving the aspect ratio, 'canvas' resizes the image canvas, padding or cropping as necessary.
    placement: places the image within the canvas
    pad_color: Padding color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.
    preserve_aspect_ratio: Border color.
    rgba_value_max: Max value for the pixel/voxel values.

    Returns
    -------
    Image with inset inserted.

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # Move to backend:
    image = Backend.to_backend(image)

    # Normalise scale:
    if type(scale) == int or type(scale) == float:
        scale = (scale,) * (inset_image.ndim - 1) + (1,)

    scale = tuple(scale)
    if len(scale) == inset_image.ndim - 1:
        scale = scale + (1,)

    # Scale inset image:
    inset_image = sp.ndimage.zoom(inset_image, zoom=scale)

    # Add border:
    inset_image = add_border(inset_image,
                             width=border_width,
                             color=border_color,
                             over_image=border_over_image,
                             rgba_value_max=rgba_value_max,
                             )

    # Check that if after scaling, the inset image is smaller than the base image:
    if any(u > v for u, v in zip(inset_image.shape, image.shape)):
        raise ValueError(f"Inset image {inset_image.shape} too large compared to base image {image.shape}.")

    # Check that both images have the same number of dimensions:
    if inset_image.ndim != image.ndim:
        raise ValueError(f"Inset image number of dimensions: {inset_image.ndim} does not match base image number of dimensions: {image.ndim}.")

    # Pad inset image:
    pad_width = tuple((v - u for u, v in zip(inset_image.shape, image.shape)))
    pad_width = pad_width[0:-1] + (0,)
    pad_width = tuple((0, p) for p in pad_width)
    padded_inset_image = xp.pad(inset_image, pad_width=pad_width)

    # Roll inset image to place it at the correct position:
    axis = tuple(range(image.ndim))
    shift = tuple((0,) * image.ndim)
    if translation is None:
        # Nothing to do...
        pass
    elif type(translation) == str:
        shift = list(shift)
        if 'top' in translation:
            shift[0] += 0
        elif 'bottom' in translation:
            shift[0] += image.shape[0] - inset_image.shape[0]
        if 'left' in translation:
            shift[1] += 0
        elif 'right' in translation:
            shift[1] += image.shape[1] - inset_image.shape[1]
        shift = tuple(shift)
    elif type(translation) == tuple:
        shift = translation + (0,)
    else:
        raise ValueError(f"Unsupported translation: {translation}")
    padded_inset_image = xp.roll(padded_inset_image, axis=axis, shift=shift)

    # Blend images together:
    result = blend_color_images(images=(image, padded_inset_image),
                                alphas=(1, alpha),
                                modes=('max', mode),
                                background_color=background_color,
                                rgba_value_max=rgba_value_max)

    return result
