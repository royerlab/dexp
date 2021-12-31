from dexp.utils import xpArray
from dexp.utils.backends import Backend


def gray2rgba(image: xpArray, alpha=None, rgba_value_max: float = 255):
    """Create a RGBA representation of a gray-level image.

    Parameters
    ----------
    image : array_like
        Input image.
    alpha : array_like, optional
        Alpha channel of the output image. It may be a scalar or an
        array that can be broadcast to ``image``. If not specified it is
        set to the maximum limit corresponding to the ``image`` dtype.

    Returns
    -------
    rgba : ndarray
        RGBA image. A new dimension of length 4 is added to input
        image shape.
    """

    xp = Backend.get_xp_module()

    arr = xp.asarray(image)

    if alpha is None:
        alpha = rgba_value_max

    rgba = xp.empty(arr.shape + (4,), dtype=arr.dtype)
    rgba[..., :3] = arr[..., xp.newaxis]
    rgba[..., 3] = alpha

    return rgba
