from typing import Union, Tuple

from dexp.processing.backends.backend import Backend
from dexp.processing.render.blend import blend_images


def insert_image(image,
                 inset_image,
                 scale: Union[float, Tuple[float, ...]] = 1,
                 position: Union[str, Tuple[int, int]] = None,
                 blend_mode: str = 'max',
                 alpha: float = 1,
                 rgb: bool = True):
    """
    Inserts an inset image into a base image.
    After scaling the inset image must be smaller than the base image.

    Parameters
    ----------
    image: Base image.
    inset_image: Inset image to place in base image.
    scale: scale factor for inset image -- scaling happens before translation.
    position: position of the inset in pixels in natural order: (x, y).
    blend_mode: blending mode.
    alpha: inset transparency.
    rgb: images are RGB images with the last dimension of length 3 or 4.

    Returns
    -------
    Image with inset inserted.

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # Normalise scale:
    if type(scale) == int or type(scale) == float:
        if rgb:
            scale = (scale,) * (inset_image.ndim - 1) + (1,)
        else:
            scale = (scale,) * inset_image.ndim
    scale = tuple(scale)
    if len(scale) == inset_image.ndim - 1:
        scale = scale + (1,)

    # Scale inset image:
    inset_image = sp.ndimage.zoom(inset_image, zoom=scale)

    # Check that if after scaling, the inset image is smaller than the base image:
    if any(u > v for u, v in zip(inset_image.shape, image.shape)):
        raise ValueError(f"Inset image {inset_image.shape} too large compared to base image {image.shape}.")

    # Check that both images have the same number of dimensions:
    if inset_image.ndim != image.ndim:
        raise ValueError(f"Inset image number of dimensions: {inset_image.ndim} does not match base image number of dimensions: {image.ndim}.")

    # Pad inset image:
    pad_width = tuple((v - u for u, v in zip(inset_image.shape, image.shape)))
    pad_width = pad_width[0:-1] + (0,) if rgb else pad_width
    pad_width = tuple((0, p) for p in pad_width)
    padded_inset_image = xp.pad(inset_image, pad_width=pad_width)

    # Roll inset image to place it at the correct position:
    axis = tuple(range(image.ndim))
    shift = tuple((0,) * image.ndim)
    if type(position) == str:
        shift = list(shift)
        if 'top' in position:
            shift[0] += 0
        elif 'bottom' in position:
            shift[0] += image.shape[0] - inset_image.shape[0]
        if 'left' in position:
            shift[1] += 0
        elif 'right' in position:
            shift[1] += image.shape[1] - inset_image.shape[1]
        shift = tuple(shift)
    elif type(position) == tuple:
        shift = position
    else:
        raise ValueError(f"Unsupported position: {position}")
    padded_inset_image = xp.roll(padded_inset_image, axis=axis, shift=shift)

    # Blend images together:
    result = blend_images(images=(image, padded_inset_image),
                          alphas=(1, alpha),
                          mode=blend_mode)

    return result
