from typing import Callable, Union

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def rgb_colormap(image: xpArray, cmap: Union[str, Callable] = None, bytes: bool = False, internal_dtype=None):
    """
    Takes an image and returns a new image of same shape + an RGB dimension.
    Colors are determined by a provided colormap.


    Parameters
    ----------
    image : Image to project
    cmap: Color map to use, can be a string or a cmap object
    bytes: If true the returned RGB values will be bytes (uint8) between 0 and 255
    internal_dtype: dtype for internal computation

    Returns
    -------
    RGB image of shape (..., 3 or 4).

    """

    xp = Backend.get_xp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    # move image to current backend:
    image = Backend.to_backend(image, dtype=internal_dtype)

    # set default cmap:
    if cmap is None:
        cmap = "viridis"

    # normalise color map:
    cmap = _normalise_colormap(cmap)

    if type(cmap) == ListedColormap or type(cmap) == LinearSegmentedColormap:
        listed_cmap: ListedColormap = cmap
        # force initialisation:
        listed_cmap(0)
        rgb_image = _apply_lut(image, listed_cmap._lut, bytes=bytes, N=listed_cmap.N, internal_dtype=internal_dtype)
    else:
        raise NotImplementedError(f"rgb_colormap not yet implemented for cmap type: {type(cmap)}")

    return rgb_image


def _normalise_colormap(cmap):
    if type(cmap) == str:
        cmap = "cet_rainbow" if cmap == "rainbow" else cmap
        cmap = "cet_bmy" if cmap == "bmy" else cmap
        from colorcet import rgb_to_hex

        rgb_to_hex(0, 0, 0)  # this is a dummy call to prevent elimination of the colorcet import by IDEs
        cmap = colormaps.get_cmap(cmap)
    elif type(cmap) == LinearSegmentedColormap or type(cmap) == ListedColormap:
        # all good here...
        cmap = cmap
    else:
        raise ValueError(f"Unknown colormap: {cmap}")
    return cmap


def _apply_lut(X, lut, bytes: bool = False, N: int = 256, internal_dtype=None):
    """
    Parameters
    ----------
    X : float or int, ndarray or scalar
        The data value(s) to convert to RGBA.
        For floats, X should be in the interval ``[0.0, 1.0]`` to
        return the RGBA values ``X*100`` percent along the Colormap line.
        For integers, X should be in the interval ``[0, Colormap.N)`` to
        return RGBA values *indexed* from the Colormap with index ``X``.
    bytes : bool
        If False (default), the returned RGBA values will be floats in the
        interval ``[0, 1]`` otherwise they will be uint8s in the interval
        ``[0, 255]``.
    N: number of entries in lut.

    Returns
    -------
    Tuple of RGBA values if X is scalar, otherwise an array of
    RGBA values with a shape of ``X.shape + (4, )``.
    """
    xp = Backend.get_xp_module()

    # move image to current backend:
    lut = Backend.to_backend(lut, dtype=internal_dtype)

    xa = xp.array(X, copy=True)
    if not xa.dtype.isnative:
        xa = xa.byteswap().newbyteorder()  # Native byteorder is faster.
    if xa.dtype.kind == "f":
        xa *= N
        # Negative values are out of range, but astype(int) would
        # truncate them towards zero.
        xa[xa < 0] = -1
        # xa == 1 (== N after multiplication) is not out of range.
        xa[xa == N] = N - 1
        # Avoid converting large positive values to negative integers.
        xp.clip(xa, -1, N, out=xa)
        xa = xa.astype(int)

    if bytes:
        lut = (lut * 255).astype(xp.uint8)

    rgba = lut[xa]

    return rgba
