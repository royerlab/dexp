from typing import Tuple, Union

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def crop_resize_pad_color_image(
    image: xpArray,
    crop: Union[int, Tuple[int, ...], Tuple[Tuple[int, int], ...]] = None,
    resize: Tuple[int, ...] = None,
    resize_order: int = 3,
    resize_mode: str = "constant",
    pad_width: Tuple[Tuple[int, int], ...] = None,
    pad_mode: str = "constant",
    pad_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
    rgba_value_max: float = 255,
):
    """
    Crops, resizes and then pad an RGB(A) image.


    Parameters
    ----------
    image: image to resize.
    crop: Crop image by removing a given number of pixels/voxels per axis. For example: ((10,20),(10,20))
        crops 10 pixels on the left for axis 0, 20 pixels from the right of axis 0, and the same for axis 2.
    resize: After cropping, the image is resized to the given shape. If any entry in the tuple is -1 then that
        position in the shape is automatically determined based on the existing shape to preserve aspect ratio.
    resize_order: The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    resize_mode: optional The mode parameter determines how the input array is extended beyond its boundaries.
        Can be: ‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’.
    pad_width: After cropping and resizing, padding is performed.
        The provided tuple is interpreted similarly to cropping.
    pad_mode: Padding mode, see numpy.pad for the available modes.
    pad_color: Padding color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.
    rgba_value_max: max value for rgba values.

    Returns
    -------
    Cropped, resized, and padded image.

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # Move to backend:
    image = Backend.to_backend(image)

    # Normalisation of crop parameter:
    if crop is not None:

        if type(crop) is int:
            crop = (crop,) * (image.ndim - 1)
        if type(crop[0]) is int:
            crop = tuple((c, c) for c in crop)

        # build the slice object to crop the image
        slicing = tuple(slice(l if l > 0 else None, -r if r > 0 else None) for l, r in crop) + (slice(None),)

        # Crop:
        image = image[slicing]

    # Normalise resize:
    if resize is not None:

        # computing resize factors:
        factors = tuple(ns / s for ns, s in zip(resize, image.shape[:-1]))

        # find all non negative factors:
        factors_no_negatives = tuple(factor for factor in factors if factor > 0)

        # compute the average (most case all factors are equal!)
        avg_factor = sum(factors_no_negatives) / len(factors_no_negatives)

        # we replace the negative values with the average:
        factors = tuple((factor if factor > 0 else avg_factor) for factor in factors)

        # handle channel dim:
        factors = factors + (1,)

        # Resizing:
        image = sp.ndimage.zoom(input=image, zoom=factors, order=resize_order, mode=resize_mode)

    # Number of channels:
    nb_channels = image.shape[-1]

    # Normalise pad_width:
    if pad_width is not None:

        # Adding a colored border:
        padded_channels = []
        for channel_index in range(nb_channels):
            channel = image[..., channel_index]

            value = pad_color[channel_index] * rgba_value_max
            padded_channel = xp.pad(channel, pad_width=pad_width, mode=pad_mode, constant_values=value)
            padded_channels.append(padded_channel)

        # Stacking:
        image = xp.stack(padded_channels, axis=-1)

    return image
