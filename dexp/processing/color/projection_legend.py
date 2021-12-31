from typing import Tuple

import numpy

from dexp.processing.color.cairo_utils import get_array_for_cairo_surface
from dexp.processing.color.colormap import rgb_colormap
from dexp.utils.backends import Backend


def depth_color_scale_legend(
    cmap,
    start: float,
    end: float,
    flip: bool = False,
    number_format: str = "{:.1f}",
    font_name: str = "Helvetica",
    font_size: float = 32.0,
    title: str = "",
    color: Tuple[float, float, float, float] = (1, 1, 1, 1),
    bar_height: float = 0.15,
    size: float = 1,
    pixel_resolution: int = 512,
):
    """
    Produces a color bar legend.

    Note: if you need to specify the unit as microns, use this symbol: μm

    Parameters
    ----------
    cmap: Color map to use
    flip: Set to rue to flip the colormap
    start: Start value
    end: End value
    number_format: Format string to represent the start and end values.
    font_name: Font name.
    font_size: Font size in pixels.
    title: Title for bar legend
    color: Title color as a tuple of normalised floats: (R, G, B, A)  (values between 0 and 1).
    bar_height: Height of the color bar in relative units [0,1].
    size: Overall size factor (default: 1)
    pixel_resolution: Base resolution in pixels to color at (pixels per unit of scale).

    Returns
    -------

    """
    width = int(size * pixel_resolution)
    height = int(size * pixel_resolution)

    # First we build the depth ramp:
    depth_ramp = numpy.linspace(0, 1, num=255)
    depth_ramp = numpy.flip(depth_ramp) if flip else depth_ramp

    # get color ramp:
    color_ramp = rgb_colormap(depth_ramp, cmap=cmap, bytes=False)

    # Create a Cairo surface:
    import cairocffi as cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.scale(width, height)

    # Adjust font size to relative coordinates:
    font_size /= pixel_resolution

    # we draw the depth ramp:
    context.set_antialias(cairo.ANTIALIAS_NONE)
    delta = 1.0 / 255
    for i in range(255):
        # determine color:
        _color = color_ramp[i]
        _color = (float(_color[0]), float(_color[1]), float(_color[2]), 1.0)
        context.set_source_rgba(*_color)

        # set position:
        pos_x = i * delta
        pos_y = 0

        # draw rectangle:
        context.rectangle(pos_x, pos_y, 2 * delta, 1)
        context.fill()

    # clear top rectangle for text:
    begin_bar = 0.5 - bar_height / 2
    end_bar = 0.5 + bar_height / 2
    context.set_operator(cairo.OPERATOR_SOURCE)
    context.set_source_rgba(0, 0, 0, 0)
    context.rectangle(0, 0, 1, begin_bar)
    context.fill()
    context.rectangle(0, end_bar, 1, end_bar)
    context.fill()

    # turn on antialiasing again for text:
    context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

    # tirn back on alpha blending:
    context.set_operator(cairo.OPERATOR_OVER)

    # draw text
    context.set_source_rgba(*color)
    context.select_font_face(font_name, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(font_size)

    text_height = context.text_extents("X")[3]

    start_text = number_format.format(start)
    context.move_to(0.01, end_bar + text_height / 2 + text_height)
    context.show_text(start_text)

    ext = context.text_extents(f"{title}")
    utw = ext[2]
    context.move_to(0.5 - utw / 2, begin_bar - text_height / 2)
    context.show_text(f"{title}")

    end_text = number_format.format(end)
    ext = context.text_extents(end_text)
    utw = ext[2]
    context.move_to(0.99 - utw, end_bar + text_height / 2 + text_height)
    context.show_text(end_text)

    # We remember where does the figure start and end vertically:
    vert_start = begin_bar - 3 * text_height / 2
    vert_end = end_bar + text_height / 2 + text_height

    surface_array = get_array_for_cairo_surface(surface)

    # Crop final image:
    height = surface_array.shape[0]
    crop_top = int((vert_start - text_height / 2) * height)
    crop_bottom = int((vert_end + text_height / 2) * height)
    surface_array = surface_array[crop_top - 10 : crop_bottom + 10, ...]

    # Move to backend:
    surface_array = Backend.to_backend(surface_array, force_copy=True)

    return surface_array
