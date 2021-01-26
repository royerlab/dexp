import numpy
from arbol import asection
from skimage.data import binary_blobs
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def generate_nuclei_background_data(length_xy=320,
                                    length_z_factor=4,
                                    zoom=1,
                                    add_noise=True,
                                    background_stength=0.2,
                                    background_scale=0.5,
                                    independent_haze=False,
                                    sphere: bool = False,
                                    add_offset=True,
                                    dtype=numpy.float16):
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

    Returns
    -------

    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if type(Backend.current()) is NumpyBackend:
        dtype = numpy.float32

    with asection("generate blob images"):
        image_gt = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype(dtype)

        if independent_haze:
            background = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=background_scale, volume_fraction=0.5).astype(dtype)
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
            distance = xp.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
            mask = distance < 0.8
            # f = 0.5*(1 + level**3 / (1 + xp.absolute(level)**3))

            image_gt *= mask

    with asection("prepare high/low image pair"):
        if background_stength > 0:
            background = sp.ndimage.gaussian_filter(background, sigma=15)
            background = background / xp.max(background)

    with asection("generate two views via blending"):
        if background_stength > 0:
            image = background_stength * background
            image += image_gt
        else:
            image = image_gt

    if length_z_factor != 1 or zoom != 1:
        with asection("downscale along z"):
            image_gt = sp.ndimage.zoom(image_gt, zoom=(zoom / length_z_factor, zoom, zoom), order=1)
            background = sp.ndimage.zoom(background, zoom=(zoom / length_z_factor, zoom, zoom), order=1)
            image = sp.ndimage.zoom(image, zoom=(zoom / length_z_factor, zoom, zoom), order=1)

    if add_noise:
        with asection("add noise"):
            image = Backend.to_numpy(image)
            image = random_noise(image, mode='speckle', var=0.5)
            image = Backend.to_backend(image)

    with asection("scale image intensities"):
        zero_level = (1 if add_offset else 0) * xp.random.uniform(95, 95 + (10 if add_noise else 0), size=image_gt.shape)
        zero_level = zero_level.astype(dtype, copy=False)
        image *= 300
        image += zero_level

    return image_gt.astype(dtype, copy=False), \
           background.astype(dtype, copy=False), \
           image.astype(dtype, copy=False),
