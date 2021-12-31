from typing import Any, Generator, Sequence, Tuple, Union

import numpy

from dexp.utils.backends import Backend


def blend_color_images(
    images: Union[Generator[Any, Any, None], Sequence[Any]],
    modes: Union[str, Sequence[str]] = "max",
    alphas: Sequence[float] = None,
    background_color: Tuple[float, float, float, float] = (0, 0, 0, 0),
    rgba_value_max: float = 255,
    internal_dtype=numpy.float32,
):
    """
    Blends multiple images together according to a blend mode

    Parameters
    ----------
    images: Sequence of images to blend.
    modes: Blending modes. Either one for all images, or one per image in the form of a sequence.
        Blending modes are: 'mean', 'add', 'satadd', 'max', 'alpha'
    alphas: Optional sequence of alpha values to use for blending, must be of same length as
        the sequence of images, optional.
    background_color: Background color as tuple of normalised floats:  (R,G,B,A). Default is transparent black.
    rgba_value_max: Max value for the pixel/voxel values.
    internal_dtype: dtype for internal computation

    Returns
    -------
    Blended image

    """

    xp = Backend.get_xp_module()

    # convert to list of images, and move to backend:
    images = list(Backend.to_backend(image) for image in images)

    # Check that there is at least one image in the image list:
    if len(images) == 0:
        raise ValueError("Blending requires at least one image!")

    # original dtype:
    original_dtype = images[0].dtype

    # Verify that all images are of the same shape:
    for image in images:
        if images[0].ndim != image.ndim or images[0].shape[:-1] != image.shape[:-1]:
            raise ValueError("All images in sequence must have the same number of dimensions and shape!")

    # verify that the images are 2D RGB(A) images:
    if images[0].ndim != 3 or not images[0].shape[2] in (3, 4):
        raise ValueError("Images must be 2D RGB(A) images!")

    # Check that the number of images and alphas match:
    if len(images) != len(alphas):
        raise ValueError(
            f"The number of images ({len(images)}) is not equal to the number of alpha values ({len(images)})."
        )

    # Create image to store result:
    result = xp.zeros_like(_ensure_rgba(images[0], rgba_value_max), dtype=internal_dtype)

    # Fill with background color:
    result[..., :] = xp.asarray(background_color, dtype=internal_dtype)

    # Blend in the images:
    for image, mode, alpha in zip(images, modes, alphas):
        # move image to backend:
        image = Backend.to_backend(image, dtype=internal_dtype)

        image = _ensure_rgba(image, rgba_value_max)

        # normalise image:
        image /= rgba_value_max
        result = _blend_function(alpha * image, result, mode)

    # scale back:
    result *= rgba_value_max

    # clip:
    result = xp.clip(result, 0, rgba_value_max)

    # convert back to original dtype (of first image)
    result = result.astype(dtype=original_dtype, copy=False)

    return result


def _ensure_rgba(image, value_max):
    xp = Backend.get_xp_module()
    # make sure that the image has an alpha channel:
    if image.shape[-1] == 3:
        pad_width = ((0, 0),) * (image.ndim - 1) + ((0, 1),)
        image = xp.pad(image, pad_width=pad_width)
        image[..., -1] = value_max
    return image


def _blend_function(image_u, image_v, mode):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if mode == "mean":
        result = 0.5 * (image_u + image_v)

    elif mode == "satadd":
        # TODO: this fails but it is unclear why:
        result = sp.special.erf(image_u + image_v)

    elif mode == "max" or mode == "rgbamax":
        result = xp.maximum(image_u, image_v)

    elif mode == "min" or mode == "rgbmin":
        result = xp.minimum(image_u, image_v)
        result[..., 3] = xp.maximum(image_u[..., 3], image_v[..., 3])

    elif image_u.shape[-1] == 4 and image_v.shape[-1] == 4:
        # See: https://en.wikipedia.org/wiki/Alpha_compositing
        # See: https://www.cairographics.org/operators/
        result = xp.zeros_like(image_u)

        src_rgb = image_u[..., 0:3]
        src_alpha = image_u[..., 3][..., xp.newaxis]
        dst_rgb = image_v[..., 0:3]
        dst_alpha = image_v[..., 3][..., xp.newaxis]

        if mode == "clear":
            # Where the second object is drawn, the first is completely removed. Anywhere else it is left intact.
            # The second object itself is not drawn.
            out_alpha = 0
            out_rgb = 0
        elif mode == "source":
            # The second object is drawn as if nothing else were below. Only outside of the blue rectangle
            # the red one is left intact.
            out_alpha = src_alpha
            out_rgb = src_rgb
        elif mode == "over" or mode == "alpha":
            # The image shows what you would expect if you held two semi-transparent slides on top of each other.
            # This operator is cairo's default operator.
            out_alpha = src_alpha + dst_alpha * (1 - src_alpha)
            out_rgb = (src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / (out_alpha + 1e-8)
        elif mode == "in":
            # The first object is removed completely, the second is only drawn where the first was.
            out_alpha = src_alpha * dst_alpha
            out_rgb = src_rgb
        elif mode == "out":
            # The blue rectangle is drawn only where the red one wasn't. Since the red one was partially
            # transparent, you can see a blue shadow in the overlapping area. Otherwise,
            # the red object is completely removed.
            out_alpha = src_alpha * (1 - dst_alpha)
            out_rgb = src_rgb
        elif mode == "atop":
            # This leaves the first object mostly intact, but mixes both objects in the overlapping area.
            # The second object object is not drawn except there.
            # If you look closely, you will notice that the resulting color in the overlapping area is different from
            # what the OVER operator produces. Any two operators produce different output in the overlapping area!
            out_alpha = dst_alpha
            out_rgb = src_alpha * src_rgb + dst_rgb * (1 - src_alpha)
        elif mode == "dest":
            # Leaves the first object untouched, the second is discarded completely.
            out_alpha = dst_alpha
            out_rgb = dst_rgb
        elif mode == "dest_over":
            # The result is similar to the OVER operator. Except that the "order" of the objects is reversed,
            # so the second is drawn below the first.
            out_alpha = (1 - dst_alpha) * src_alpha + dst_alpha
            out_rgb = (src_alpha * src_rgb * (1 - dst_alpha) + dst_alpha * dst_rgb) / (out_alpha + 1e-8)
        elif mode == "dest_in":
            # The blue rectangle is used to determine which part of the red one is left intact.
            # Anything outside the overlapping area is removed.
            # This works like the IN operator, but again with the second object "below" the first.
            out_alpha = src_alpha * dst_alpha
            out_rgb = dst_rgb
        elif mode == "dest_atop":
            # Same as the ATOP operator, but again as if the order of the drawing operations
            # had been reversed.
            out_alpha = src_alpha
            out_rgb = src_rgb * (1 - dst_alpha) + dst_alpha * dst_rgb
        elif mode == "xor":
            # The output of the XOR operator is the same for both bounded and unbounded source interpretations.
            out_alpha = src_alpha + dst_alpha - 2 * src_alpha * dst_alpha
            out_rgb = src_alpha * src_rgb * (1 - dst_alpha) + dst_alpha * dst_rgb * (1 - src_alpha)
            out_rgb /= out_alpha + 1e-8
        elif mode == "add":
            # The output of the ADD operator is the same for both bounded and unbounded source interpretations.
            out_alpha = xp.minimum(1, src_alpha + dst_alpha)
            out_rgb = (src_alpha * src_rgb + dst_alpha * dst_rgb) / (out_alpha + 1e-8)
        elif mode == "saturate":
            # The output of the ADD operator is the same for both bounded and unbounded source interpretations.
            out_alpha = xp.minimum(1, src_alpha + dst_alpha)
            out_rgb = (xp.minimum(src_alpha, 1 - dst_alpha) * src_rgb + dst_alpha * dst_rgb) / (out_alpha + 1e-8)
        else:
            raise ValueError(
                f"Invalid alpha blending mode {mode} or incompatible images: {image_u.shape} {image_v.shape}"
            )

        result[..., 0:3] = out_rgb.squeeze()
        result[..., 3] = out_alpha.squeeze()
    else:
        raise ValueError(f"Invalid blending mode {mode} or incompatible images: {image_u.shape} {image_v.shape}")

    return result
