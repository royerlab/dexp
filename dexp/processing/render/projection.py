from typing import Union, Tuple, Callable

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.colormap import rgb_colormap, _normalise_colormap
from dexp.processing.render.insert import insert_image
from dexp.processing.render.projection_legend import depth_color_scale_legend
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
                dlim: Tuple[float, float] = None,
                depth_stabilisation: bool = False,
                rgb_gamma: float = 1,
                transparency: bool = True,
                legend_size: float = 0,
                legend_depth_scale: float = 1,
                legend_depth_title: str = 'voxels',
                legend_position: Union[str, Tuple[int, int]] = 'bottom_left',
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
    dlim: Depth limits. For example, a value of (0.1, 0.7) means that the colormap start at a normalised depth of 0.1, and ends at a normalised depth of 0.7, other values are clipped.
    depth_stabilisation: Uses the center of mass calculation to shift the center of the depth color map to teh center of mass of the image content.
    rgb_gamma: Gamma correction applied to the resulting RGB image.
    transparency: If True, the best attempt is made to use alpha transparency in the resulting image. Not available in all modes. RGB output is not 'alpha pre-multiplied'.
    legend_size: Multiplicative factor to control size of legend. If 0, no legend is generated.
    legend_depth_scale: Float that gives the scale in some unit of each voxel (along the projection direction). Only in color projection modes.
    legend_depth_title: title for the color-coded depth legend.
    internal_dtype : dtype for internal computation

    Returns
    -------
    Projected image of shape (..., 3 or 4), legend image of shape (..., 3 or 4)

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
        elif mode == 'maxcolor':
            cmap = 'cet_bmy'
        elif mode == 'colormax':
            cmap = 'cet_bmy'

    # normalise color map:
    cmap = _normalise_colormap(cmap)

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

        expanded_indices = xp.expand_dims(indices, axis=axis)
        values = xp.take_along_axis(image, expanded_indices, axis=axis)
        values = xp.squeeze(values)
        # values = xp.max(image, axis=axis)

        # apply color map, this is just the chroma-coding of depth
        norm_factor = xp.array(1.0 / float(image.shape[axis] - 1)).astype(internal_dtype, copy=False)
        normalised_depth = norm_factor * indices

        # Crude depthstabilisation, not recomended:
        if depth_stabilisation:
            com = center_of_mass(image)
            delta = (com[axis] / image.shape[axis]) - 0.5
            normalised_depth -= delta

        if dlim is not None:
            normalised_depth = _apply_depth_limits(normalised_depth, dlim)
        projection = rgb_colormap(normalised_depth, cmap=cmap, bytes=False)

        # Next we multiply the chroma-code with the intensity of the corresponding voxel:
        if transparency:
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

    if 'color' in mode and legend_size != 0:
        depth = image.shape[axis] * (abs(dlim[1] - dlim[0])) * legend_depth_scale
        legend = depth_color_scale_legend(cmap=cmap,
                                          start=0,
                                          end=depth,
                                          title=legend_depth_title,
                                          size=legend_size)

        projection = insert_image(projection,
                                  legend,
                                  position=legend_position,
                                  blend_mode='max')

    return projection


def _apply_depth_limits(depth_map,
                        dlim: Tuple[float, float]):
    xp = Backend.get_xp_module()

    min_v, max_v = dlim
    depth_map = xp.clip(depth_map, min_v, max_v)
    depth_map -= depth_map.min()
    depth_map /= depth_map.max()

    return depth_map


def _inplace_cumsum(image, axis, dir=1):
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
