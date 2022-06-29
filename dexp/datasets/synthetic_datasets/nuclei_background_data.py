from typing import Any

import numpy
from arbol import asection
from skimage.util import random_noise

from dexp.datasets.synthetic_datasets import binary_blobs
from dexp.utils.backends import Backend, NumpyBackend


@asection("Generating synthetic nuclei data")
def generate_nuclei_background_data(
    length_xy=320,
    length_z_factor=4,
    zoom=1,
    add_noise=True,
    background_strength=0.2,
    background_scale=0.5,
    independent_haze=False,
    sphere: bool = False,
    radius: float = 0.8,
    add_offset=True,
    dtype=numpy.float16,
    internal_dtype=numpy.float16,
    rng: Any = 42,
):
    """

    Parameters
    ----------
    length_xy
    length_z_factor
    zoom
    add_noise
    background_stength
    background_scale
    independent_haze
    sphere
    add_offset
    dtype
    internal_dtype

    Returns
    -------

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if isinstance(rng, int):
        rng = xp.random.RandomState(seed=rng)

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = xp.float32

    with asection("generate blob images"):
        image_gt = binary_blobs(
            length=length_xy,
            n_dim=3,
            blob_size_fraction=0.07,
            volume_fraction=0.1,
            rng=rng,
        ).astype(internal_dtype)

        if independent_haze:
            background = binary_blobs(
                length=length_xy,
                n_dim=3,
                blob_size_fraction=background_scale,
                volume_fraction=0.5,
                rng=rng,
            ).astype(internal_dtype)

        else:
            background = image_gt.copy()

    with asection("convert blob images to backend"):
        image_gt = Backend.to_backend(image_gt)
        background = Backend.to_backend(background)

    if sphere:
        with asection("sphere mask"):
            lz, ly, lx = image_gt.shape
            x = xp.linspace(-1, 1, num=lx)
            y = xp.linspace(-1, 1, num=ly)
            z = xp.linspace(-1, 1, num=lz)
            xx, yy, zz = numpy.meshgrid(x, y, z, sparse=True)
            distance = xp.sqrt(xx**2 + yy**2 + zz**2)
            mask = distance < radius
            # f = 0.5*(1 + level**3 / (1 + xp.absolute(level)**3))

            image_gt *= mask

    with asection("prepare high/low image pair"):
        if background_strength > 0:
            background = sp.ndimage.gaussian_filter(background, sigma=15)
            background = background / xp.max(background)

    with asection("generate two views via blending"):
        if background_strength > 0:
            image = background_strength * background
            image += image_gt
        else:
            image = image_gt

    if length_z_factor != 1 or zoom != 1:
        with asection("upscale all dimensions and downscale along z"):
            image_gt = sp.ndimage.zoom(image_gt, zoom=(zoom / length_z_factor, zoom, zoom), order=1)
            background = sp.ndimage.zoom(background, zoom=(zoom / length_z_factor, zoom, zoom), order=1)
            image = sp.ndimage.zoom(image, zoom=(zoom / length_z_factor, zoom, zoom), order=1)

    if add_noise:
        with asection("add noise"):
            image = Backend.to_numpy(image)
            image = random_noise(image, mode="speckle", var=0.5, seed=123)
            image = Backend.to_backend(image)

    with asection("scale image intensities"):
        zero_level = (1 if add_offset else 0) * rng.uniform(95, 95 + (10 if add_noise else 0), size=image_gt.shape)
        zero_level = zero_level.astype(image.dtype, copy=False)
        image *= 300
        image += zero_level
        image = xp.clip(image - 1, 0, None, out=image)

    return (
        image_gt.astype(dtype, copy=False),
        background.astype(dtype, copy=False),
        image.astype(dtype, copy=False),
    )
