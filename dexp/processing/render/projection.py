from typing import Union, Tuple

from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.utils.normalise import normalise_functions


def rgb_project(image,
                axis: int = 0,
                dir: int = -1,
                mode: str = 'max',
                attenuation: float = 0,
                gamma: float = 1,
                clim: Tuple[float, float] = None,
                cmap: Union[str, str] = None,
                attenuation_filtering: float = 4,
                internal_dtype=None):
    """
    Projects an image along a given axis given a specified method (max projection, max projection color-coded depth, ...)
    and produces a rendered RGB image. This function offers features similar to volume rendering:
    attenuation, projection direction direction, ...


    Parameters
    ----------
    image : Image to project
    axis : axis along which to project
    dir : projection diretion, can be either '-1' for top to botttom or '1' for bottom to top -- assuming top corresponds to the positive direction of the projection axis.
    mode : projection mode, can be: 'max' for max projection, 'maxcolor' for max color projection, ...
    attenuation : How much to attenuate when projecting.
    gamma: Gamma correction to apply
    clim : color limits for applying the colormap.
    attenuation_filtering: standard deviation of the gaussian filter used to preprocess the image for the purpose of computing the attenuation map.
    Important: this does not affect sharpness of the final image, it only affects the sharpness of the attenuation itself.
    cmap: Color map to use, can be a string or a cmap object
    internal_dtype : dtype for internal computation

    Returns
    -------
    Projected image of shape (..., 3 or 4).

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = image.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    # move image to current backend:
    image = Backend.to_backend(image, dtype=internal_dtype)

    # set default cmap:
    if cmap is None:
        if mode == 'max':
            cmap = 'viridis'
        elif mode == 'color':
            cmap = 'viridis'
        elif mode == 'colormax':
            cmap = 'cet_bmy'

    # normalise color map:
    if type(cmap) == str:
        from colorcet import rgb_to_hex
        rgb_to_hex(0, 0, 0)  # this is a dummy call to prevent eliminationof the colorcet import by IDEs
        cmap = get_cmap(cmap)
    elif type(cmap) == LinearSegmentedColormap:
        # all good here...
        cmap = cmap
    else:
        raise ValueError(f"Unknown colormap: {cmap}")

    # Normalise
    norm_fun, _ = normalise_functions(image, quantile=0.0001, minmax=clim, clip=True)
    image = norm_fun(image)

    # Apply gamma:
    if gamma != 1:
        image **= gamma

    if attenuation != 0:

        if attenuation_filtering > 0:
            image_for_attenuation = sp.ndimage.gaussian_filter(image, sigma=attenuation_filtering)
        else:
            image_for_attenuation = image

        if dir == -1:
            cum_density = xp.cumsum(image_for_attenuation, axis=axis)
        elif dir == +1:
            image_flipped = xp.flip(image_for_attenuation, axis=axis)
            cum_density = xp.cumsum(image_flipped, axis=axis)
            cum_density = xp.flip(cum_density, axis=axis)
        else:
            raise ValueError(f"Invalid direction: {dir}, must be '-1' or '+1' ")

        image = image * xp.exp(-attenuation * cum_density)

    # Perform projection
    if mode == 'max':
        # max projection:
        projection = xp.max(image, axis=axis)

        # apply color map:
        projection = cmap(projection)

    elif mode == 'colormax':

        # argmax:
        indices = xp.argmax(image, axis=axis)
        values = xp.max(image, axis=axis)

        # apply color map, this is just the chroma-coding of depth
        norm_factor = 1.0 / float(image.shape[axis] - 1)
        projection = cmap(norm_factor * indices)
        dtype = projection.dtype

        # Next we multiply the chroma-code with the inetnsity of the corresponding voxel:
        projection[..., 0:3] *= values[..., xp.newaxis]
        projection = projection.astype(dtype, copy=False)


    else:
        raise ValueError(f"Invalid projection mode: {mode}")

    return projection
