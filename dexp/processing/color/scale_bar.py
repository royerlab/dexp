from typing import Tuple, Union

from dexp.processing.color.blend import blend_color_images
from dexp.processing.color.cairo_utils import get_array_for_cairo_surface
from dexp.utils import xpArray
from dexp.utils.backends import Backend


def insert_scale_bar(
    image: xpArray,
    length_in_unit: float = 1,
    pixel_scale: float = 1,
    bar_height: int = 4,
    margin: float = 1,
    translation: Union[str, Tuple[Union[int, float], ...]] = "bottom_right",
    color: Tuple[float, float, float, float] = None,
    number_format: str = "{:.1f}",
    font_name: str = "Helvetica",
    font_size: float = 32,
    unit: str = "μm",
    mode: str = "max",
):
    """
    Inserts a scale bar into an image.

    Parameters
    ----------
    image: Image into which to insert the scale bar.
    length_in_unit: Length of scale bar in the provided unit.
    pixel_scale: conversion factor from pixels to units -- what is the side length of a pixel/voxel in units.
    bar_height: Height of th scale bar in pixels
    margin: margin around bar expressed in units relative to the text height
    translation: Positions of the scale bar in pixels in numpy order: (y, x).
        Can also be a string: 'bottom_left', 'bottom_right', 'top_left', 'top_right'.
    color: Color of the bar and text as tuple of 4 values: (R, G, B, A)
    number_format: Format string to represent the start and end values.
    font_name: Font name.
    font_size: Font size in pixels.
    unit: Unit name.
    mode: Blending mode. See function 'blend_color_images' for available blending modes.

    Returns
    -------
    Image with inserted scale bar.

    """

    # Move to backend:
    image = Backend.to_backend(image)

    if color is None:
        color = (1, 1, 1, 1)

    # Image shape:
    height, width = image.shape[:-1]

    # Bar length in pixels/voxels:
    bar_length = length_in_unit / pixel_scale

    # Replace um with µm:
    if unit.strip() == "um":
        unit = "µm"

    scale_bar_array = _generate_scale_bar_image(
        width,
        height,
        translation,
        bar_height,
        bar_length,
        margin,
        color,
        font_name,
        font_size,
        length_in_unit,
        number_format,
        unit,
    )

    # Blend images:
    result = blend_color_images(images=(image, scale_bar_array), alphas=(1, 1), modes=("max", mode))

    return result, scale_bar_array


def _generate_scale_bar_image(
    width,
    height,
    translation,
    bar_height,
    bar_length,
    margin,
    color,
    font_name,
    font_size,
    length_in_unit,
    number_format,
    unit,
):
    # Create surface:
    import cairocffi as cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.scale(1, 1)

    # Configure text rendering:
    context.select_font_face(font_name, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size)

    # text to be rendered:
    text = f"{number_format.format(length_in_unit)} {unit}"

    # determine text height and width:
    text_width = context.text_extents(text)[2]
    text_height = context.text_extents(text)[3]

    # Margins:
    margin_height = margin * text_height
    margin_width = margin * text_height

    # First we turn off antialiasing, works better:
    context.set_antialias(cairo.ANTIALIAS_NONE)

    # Turn off alpha blending and clears the image:
    context.set_operator(cairo.OPERATOR_SOURCE)
    context.set_source_rgba(0, 0, 0, 0)
    context.rectangle(0, 0, 1, 1)
    context.fill()

    # We set the color for the bar:
    context.set_source_rgba(*color)

    # Set the position of the bar, that is the left-most end of it, including the margin:
    if type(translation) == str:
        if "top" in translation:
            y = margin_height
        elif "bottom" in translation:
            y = height - (margin_height + bar_height)
        if "left" in translation:
            x = margin_width
        elif "right" in translation:
            x = width - (margin_width + bar_length / 2 + max(bar_length / 2, text_width / 2))
    elif type(translation) == tuple:
        y, x = translation
    else:
        raise ValueError(f"Invalid translation: {translation}")

    # We draw the scale bar itself:
    context.rectangle(x, y - bar_height / 2, bar_length, bar_height)
    context.fill()

    # Turn on antialiasing again for text:
    context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

    # Turn back on alpha blending:
    context.set_operator(cairo.OPERATOR_OVER)

    # Set text color:
    context.set_source_rgba(*color)

    # draw text at correct location:
    if "top" in translation:
        context.move_to(x + bar_length / 2 - text_width / 2, y + (3 * text_height / 2))
    else:
        context.move_to(x + bar_length / 2 - text_width / 2, y - text_height / 2)
    context.show_text(text)

    # Get array from surface:
    surface_array = get_array_for_cairo_surface(surface)
    return surface_array
