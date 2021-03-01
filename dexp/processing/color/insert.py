from typing import Union, Tuple, Sequence

from dexp.processing.backends.backend import Backend
from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.border import add_border


def insert_color_image(image,
                       inset_image,
                       scale: Union[float, Tuple[float, ...]] = 1,
                       translation: Union[str, Sequence[Tuple[Union[int, float], ...]]] = None,
                       border_width: int = 0,
                       border_color: Tuple[float, float, float, float] = None,
                       border_over_image: bool = False,
                       mode: str = 'max',
                       alpha: float = 1):
    """
    Inserts an inset image into a base image.
    After scaling the inset image must be smaller than the base image.

    Parameters
    ----------
    image: Base image.
    inset_image: Inset image to place in base image.
    scale: scale factor for inset image -- scaling happens before translation.
    translation: positions of the insets in pixels in natural order: (x, y). Can also be a string: 'bottom_left', 'bottom_right', 'top_left', 'top_right'.
    width: Width of border added to insets.
    color: Border color.
    over_image: If True the border is not added but overlayed over the image, the image does not change size.
    mode: blending mode.
    alpha: inset transparency.

    Returns
    -------
    Image with inset inserted.

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

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
                             over_image=border_over_image
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
                                modes=('max', mode))

    return result
