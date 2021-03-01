from typing import Tuple

from dexp.processing.backends.backend import Backend


def add_border(image,
               width: int = 2,
               color: Tuple[float, float, float, float] = None,
               over_image: bool = False,
               rgb_max_value: int = 255
               ):
    """
    Adds a color border to an image

    Parameters
    ----------
    image: Base image.
    width: Width of border.
    color: Border color.
    over_image: If True the border is not added but overlayed over the image, the image does not change size.
    rgb_max_value: max valiue for rgba values.

    Returns
    -------
    Image with border added or overlayed

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # If border is 0 then rwe return the image unchanged:
    if width == 0:
        return image

    # Default color:
    if color is None:
        color = (1, 1, 1, 1)

    # Bring alpha to the front:
    image = image.transpose(2, 0, 1)

    shape = image.shape

    # Padding
    pad_width = ((width, width), (width, width))

    # New image with border:
    new_shape = shape if over_image else (shape[0],) + tuple(s + 2 * width for s in shape[1:])
    image_with_border = xp.zeros(shape=new_shape, dtype=image.dtype)

    for i, channel in enumerate(image):
        value = int(color[i] * rgb_max_value)
        if over_image:
            channel = channel[..., width:-width, width:-width]

        padded_channel = xp.pad(channel,
                                pad_width=pad_width,
                                mode='constant',
                                constant_values=value)
        image_with_border[i,] = padded_channel

    image_with_border = image_with_border.transpose(1, 2, 0)

    return image_with_border
