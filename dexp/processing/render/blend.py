from typing import Sequence

from dexp.processing.backends.backend import Backend





def blend(images: Sequence['Array'],
          alphas: Sequence[float] = None,
          mode: str = 'over',
          rgb: bool = True):

    """
    Blends multiple images together according to a blend mode

    Parameters
    ----------
    images: Sequence of images to blend.
    alphas: Optional sequence of alpha values to use for blending, must be of same length as the sequence of images, optional.
    mode: blending mode.
    rgb: images are RGB images with the last dimension of length 3 or 4.

    Returns
    -------
    Blended image

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # convert to list of images:
    images = list(images)

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
    result = xp.zeros_like(images[0])

    # Blend in the images:
    for alpha, image in zip(alphas,images):
        image = Backend.to_backend(image)
        result = _blend_function(result, image, mode, alpha)

    return result


def _blend_function(image_u, image_v, alpha_u, alpha_v, mode):

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image_u = Backend.to_backend(image_u)
    image_v = Backend.to_backend(image_v)

    if mode == 'sum':
        result = image_u * alpha_u + image_v*alpha_v
    elif mode == 'min':
        result = xp.maximum(image_u * alpha_u, image_v * alpha_v)
    elif mode == 'max':
        result = xp.maximum(image_u * alpha_u, image_v * alpha_v)

    else:
        raise ValueError("Unknown blend mode")

    return result



