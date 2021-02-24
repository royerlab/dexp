from typing import Union, Tuple, Callable

from matplotlib.cm import get_cmap
from matplotlib.colors import LinearSegmentedColormap

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.colormap import rgb_colormap
from dexp.processing.utils.center_of_mass import center_of_mass
from dexp.processing.utils.normalise import normalise_functions


def rgb_project(image,
                axis: int = 0,
                dir: int = -1,
                mode: str = 'max',
                attenuation: float = 0,
                gamma: float = 1,
                clim: Tuple[float, float] = None,
                cmap: Union[str, Callable] = None,
                attenuation_filtering: float = 4,
                depth_gamma: float = 1,
                depth_stabilisation: bool = False,
                rgb_gamma: float = 1,
                alpha: bool = True,
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
    mode : projection mode, can be: 'max' for max projection, 'colormax' and 'maxcolor' for max color projections. Explanation: colormax applies first max and then colorises,
     maxcolor, first colorises each voxel then computes the max. This means that maxcolor is 3x more memory intensive (with current implementation).
    attenuation : How much to attenuate when projecting.
    gamma: Gamma correction to apply
    clim : color limits for applying the colormap.
    cmap: Color map to use, can be a string or a cmap object
    attenuation_filtering: standard deviation of the gaussian filter used to preprocess the image for the purpose of computing the attenuation map.
    Important: this does not affect sharpness of the final image, it only affects the sharpness of the attenuation itself.
    depth_gamma: Gamma correction applied to the stack depth to accentuate (depth_gamma < 1) color variations with depth at the center of the stack.
    depth_stabilisation: Uses the center of mass calculation to shift the center of the depth color map to teh center of mass of the image content.
    rgb_gamma: Gamma correction applied to the resulting RGB image.
    alpha: If True, the best attempt is made to use alpha transparency in the resulting image. Not available in all modes.
    internal_dtype : dtype for internal computation

    Returns
    -------
    Projected image of shape (..., 3 or 4).

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = xp.float16

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
        rgb_to_hex(0, 0, 0)  # this is a dummy call to prevent elimination of the colorcet import by IDEs
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

        if dir == -1 or dir == +1:
            cum_density = _inplace_cumsum(image_for_attenuation, axis=axis, dir=-dir)

        else:
            raise ValueError(f"Invalid direction: {dir}, must be '-1' or '+1' ")

        image *= xp.exp(-attenuation * cum_density)

    # Perform projection
    if mode == 'max':
        # max projection:
        projection = xp.max(image, axis=axis)

        # apply color map:
        projection = rgb_colormap(projection, cmap=cmap, bytes=False)

    elif mode == 'maxcolor':

        # compute a depth map of same shape as the image:
        depth = image.shape[axis]
        depth_values = xp.linspace(0, 1, num=depth) if dir < 0 else xp.linspace(1, 0, num=depth)
        depth_values = xp.expand_dims(depth_values, axis=tuple(range(image.ndim - 1)))
        depth_values = xp.moveaxis(depth_values, -1, axis)

        # Apply the colormap:
        color_ramp = rgb_colormap(depth_values, cmap=cmap, bytes=False)

        # Multiply with image leveraing broadcasting:
        color_image = color_ramp * image[..., xp.newaxis]

        # Projecting:
        projection = xp.max(color_image, axis=axis)

    elif mode == 'colormax':

        # argmax:
        indices = xp.argmax(image, axis=axis)
        values = xp.max(image, axis=axis)

        # apply color map, this is just the chroma-coding of depth
        norm_factor = xp.array(1.0 / float(image.shape[axis] - 1)).astype(internal_dtype, copy=False)
        normalised_depth = norm_factor * indices

        if depth_stabilisation:
            com = center_of_mass(image)
            delta = (com[axis] / image.shape[axis]) - 0.5
            normalised_depth -= delta

        if depth_gamma != 1.0:
            # this could be made faster, but does it matter (those are 2D images)
            normalised_depth = _apply_depth_gamma(normalised_depth, depth_gamma)
        projection = rgb_colormap(normalised_depth, cmap=cmap, bytes=False)

        # Next we multiply the chroma-code with the intensity of the corresponding voxel:
        if alpha:
            projection[..., 3] *= values
        else:
            projection[..., 0:3] *= values[..., xp.newaxis]

    else:
        raise ValueError(f"Invalid projection mode: {mode}")

    if rgb_gamma != 1.0:
        projection **= rgb_gamma

    projection *= 255
    projection = xp.clip(projection, 0, 255)
    projection = projection.astype(xp.uint8, copy=False)

    return projection


def _apply_depth_gamma(depth_map, gamma):
    if type(Backend.current()) is CupyBackend:
        import cupy

        @cupy.fuse
        def fun(depth_map, gamma):
            depth_map *= 2
            depth_map -= 1
            depth_map = cupy.sign(depth_map) * (cupy.absolute(depth_map) ** gamma)
            depth_map += 1
            depth_map *= 0.5
            return depth_map

        return fun(depth_map, gamma)

    else:

        xp = Backend.get_xp_module()
        depth_map *= 2
        depth_map -= 1
        depth_map = xp.sign(depth_map) * (xp.absolute(depth_map) ** gamma)
        depth_map += 1
        depth_map *= 0.5
        return depth_map


def _inplace_cumsum(image, axis, dir = 1):
    xp = Backend.get_xp_module()

    length = image.shape[axis]

    accumulator = xp.zeros_like(xp.take(image, 0, axis=axis))

    positions = list(range(length))

    if dir < 0:
        positions.reverse()

    for i in positions:
        # take slice:
        _slice = xp.take(image, i, axis=axis)

        # accumulate:
        accumulator += _slice

        # place sum into array:
        xp.moveaxis(image, axis, 0)[i] = accumulator

    return image
