from typing import Sequence

import numpy

from dexp.processing.backends.backend import Backend


def blend_images(images: Sequence['Array'],
                 alphas: Sequence[float] = None,
                 value_max: float = 255,
                 mode: str = 'max',
                 internal_dtype=numpy.float32):
    """
    Blends multiple images together according to a blend mode

    Parameters
    ----------
    images: Sequence of images to blend.
    alphas: Optional sequence of alpha values to use for blending, must be of same length as the sequence of images, optional.
    value_max: Max value for teh pixel/voxel values.
    mode: Blending mode. can be: 'mean', 'add', 'satadd', 'max', 'alpha'
    internal_dtype: dtype for internal computation

    Returns
    -------
    Blended image

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # convert to list of images:
    images = list(images)

    # original dtype:
    original_dtype = images[0].dtype

    # normalise alphas:
    # alpha_sum = sum(alphas)
    # alphas = list(alpha/alpha_sum for alpha in alphas)

    # Check that there is at least one image in the image list:
    if len(images) == 0:
        raise ValueError(f"Blending requires at least one image!")

    # Check that the number of images and alphas match:
    if len(images) != len(alphas):
        raise ValueError(f"The number of images ({len(images)}) is not equal to the number of alpha values ({len(images)}).")

    # Check that all images have the same shape:
    for image in images:
        if image.shape != images[0].shape:
            raise ValueError(f"Not all images have the same shape: {image.shape}.")

    # Create image to store result:
    result = xp.zeros_like(images[0], dtype=internal_dtype)

    # Blend in the images:
    for alpha, image in zip(alphas, images):
        # move image to backend:
        image = Backend.to_backend(image, dtype=internal_dtype)

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
    elif mode == 'rgbmax':
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
