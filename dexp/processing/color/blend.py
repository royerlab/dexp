from typing import Sequence, Union, Generator, Any

import numpy

from dexp.processing.backends.backend import Backend


def blend_color_images(images: Union[Generator[Any, Any, None], Sequence[Any]],
                       alphas: Sequence[float] = None,
                       value_max: float = 255,
                       modes: Union[str, Sequence[str]] = 'max',
                       internal_dtype=numpy.float32):
    """
    Blends multiple images together according to a blend mode

    Parameters
    ----------
    images: Sequence of images to blend.
    alphas: Optional sequence of alpha values to use for blending, must be of same length as the sequence of images, optional.
    value_max: Max value for the pixel/voxel values.
    modes: Blending modes. Either one for all images, or one per image in the form of a sequence. Blending modes are: 'mean', 'add', 'satadd', 'max', 'alpha'
    internal_dtype: dtype for internal computation

    Returns
    -------
    Blended image

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # convert to list of images, and move to backend:
    images = list(Backend.to_backend(image) for image in images)

    # Check that there is at least one image in the image list:
    if len(images) == 0:
        raise ValueError(f"Blending requires at least one image!")

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
        raise ValueError(f"The number of images ({len(images)}) is not equal to the number of alpha values ({len(images)}).")

    # Create image to store result:
    result = xp.zeros_like(_ensure_rgba(images[0], value_max), dtype=internal_dtype)

    # Blend in the images:
    for image, mode, alpha in zip(images, modes, alphas):
        # move image to backend:
        image = Backend.to_backend(image, dtype=internal_dtype)

        image = _ensure_rgba(image, value_max)

        # normalise image:
        image /= value_max
        result = _blend_function(result, alpha * image, mode)

    # scale back:
    result *= value_max

    # clip:
    result = xp.clip(result, 0, value_max)

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

    if mode == 'mean':
        result = 0.5 * (image_u + image_v)
    elif mode == 'add' or mode == 'addclip':
        result = image_u + image_v
    elif mode == 'satadd':
        # TODO: this fails but it is unclear why:
        result = sp.special.erf(image_u + image_v)
    elif mode == 'max' or mode == 'rgbamax':
        result = xp.maximum(image_u, image_v)
    elif mode == 'alpha' and image_u.shape[-1] == 4 and image_v.shape[-1] == 4:
        # See: https://en.wikipedia.org/wiki/Alpha_compositing

        src_rgb = image_u[..., 0:3]
        src_alpha = image_u[..., 3][..., xp.newaxis]
        dst_rgb = image_v[..., 0:3]
        dst_alpha = image_v[..., 3][..., xp.newaxis]

        result = xp.zeros_like(image_u)
        out_alpha = (src_alpha + dst_alpha * (1 - src_alpha))
        result[..., 0:3] = ((src_rgb * src_alpha + dst_rgb * dst_alpha * (1 - src_alpha)) / out_alpha).squeeze()
        result[..., 3] = out_alpha.squeeze()
    else:
        raise ValueError(f"Invalid blending mode {mode} or incompatible images: {image_u.shape} {image_v.shape}")

    return result
