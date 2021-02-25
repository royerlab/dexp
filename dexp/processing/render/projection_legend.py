import numpy

from dexp.processing.render.colormap import rgb_colormap


def depth_color_scale_legend(cmap, start, end, title, legend_size):
    """
    Produces a color bar legend
    
    Parameters
    ----------
    cmap
    start
    end
    title
    legend_size

    Returns
    -------

    """
    width = int(legend_size * 512)
    height = int(legend_size * 512)

    # First we build the depth ramp:
    depth_ramp = numpy.linspace(0, 1, num=255)

    # get color ramp:
    color_ramp = rgb_colormap(depth_ramp, cmap=cmap, bytes=False)

    # Create surface from array:
    import cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    context.scale(width, height)

    # we draw the depth ramp:
    context.set_antialias(cairo.ANTIALIAS_NONE)
    delta = 1.0 / 255
    for i in range(255):
        # determine color:
        color = color_ramp[i]
        color = (float(color[0]), float(color[1]), float(color[2]), 1.0)
        context.set_source_rgba(*color)

        # set position:
        pos_x = i * delta
        pos_y = 0

        # draw rectangle:
        context.rectangle(pos_x, pos_y, 2 * delta, 1)
        context.fill()

    # clear top rectangle for text:
    bar_height = 0.3
    begin_bar = 0.5 - bar_height / 2
    end_bar = 0.5 + bar_height / 2
    context.set_source_rgba(0, 0, 0, 1)
    context.rectangle(0, 0, 1, begin_bar)
    context.fill()
    context.rectangle(0, end_bar, 1, end_bar)
    context.fill()

    # turn on antialiasing again for text:
    context.set_antialias(cairo.ANTIALIAS_SUBPIXEL)

    # draw text
    context.set_source_rgba(1, 1, 1, 1)
    context.select_font_face("Helvetica",
                             cairo.FONT_SLANT_NORMAL,
                             cairo.FONT_WEIGHT_NORMAL)
    context.set_font_size(0.1)

    text_height = context.text_extents('X')[3]

    context.move_to(0.01, end_bar + text_height / 3 + text_height)
    context.show_text(f"{start}")

    ext = context.text_extents(f"{title}")
    utw = ext[2]
    context.move_to(0.5 - utw / 2, begin_bar - text_height / 3)
    context.show_text(f"{title}")

    end_text = f"{end}"
    ext = context.text_extents(end_text)
    utw = ext[2]
    context.move_to(0.99 - utw, end_bar + text_height / 3 + text_height)
    context.show_text(end_text)

    # Get pycairo surface buffer:
    buffer = surface.get_data()

    # Reshape array to get an extra uint8 axis:
    surface_array = numpy.ndarray(shape=(height, width, 4), dtype=numpy.uint8, buffer=buffer)

    # We have now: BGRA, we need to flip color axis because of endianness to ARGB:
    surface_array = numpy.flip(surface_array, axis=surface_array.ndim - 1)

    # Convert ARGB to RGBA:
    surface_array = numpy.roll(surface_array, shift=-1, axis=surface_array.ndim - 1)

    return surface_array.copy()
