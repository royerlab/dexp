import numpy

from dexp.utils.backends import Backend


def get_array_for_cairo_surface(surface: "ImageSurface"):  # noqa: F821
    """
    Returns an array given a ImageSurface from PyCairo.

    Parameters
    ----------
    surface : surface

    Returns
    -------
    RGBA numpy array of shape: (...,4)

    """

    from cairocffi import ImageSurface

    surface: ImageSurface

    xp = Backend.get_xp_module()

    width = surface.get_width()
    height = surface.get_height()

    # Get pycairo surface buffer:
    buffer = surface.get_data()

    # Reshape array to get an extra uint8 axis:
    surface_array = numpy.ndarray(shape=(height, width, 4), dtype=xp.uint8, buffer=buffer)

    # Move to backend:
    surface_array = Backend.to_backend(surface_array)

    # We have now: BGRA, we need to flip color axis because of endianness to ARGB:
    surface_array = xp.flip(surface_array, axis=surface_array.ndim - 1)

    # Convert ARGB to RGBA:
    surface_array = xp.roll(surface_array, shift=-1, axis=surface_array.ndim - 1)

    return surface_array
