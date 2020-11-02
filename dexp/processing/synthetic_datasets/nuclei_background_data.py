import numpy
from skimage.data import binary_blobs
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.utils.timeit import timeit


def generate_nuclei_background_data(backend:Backend,
                                  length_xy=320,
                                  length_z_factor=4,
                                  add_noise=True,
                                  background_stength=0.2,
                                  background_scale=0.5,
                                  independent_haze=False):

    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    with timeit("generate blob images"):
        image_gt = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')
        if independent_haze:
            background = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=background_scale, volume_fraction=0.5).astype('f4')
        else:
            background = image_gt.copy()

    with timeit("convert blob images to backend"):
        image_gt = backend.to_backend(image_gt)
        background = backend.to_backend(background)

    with timeit("prepare high/low image pair"):
        background = sp.ndimage.gaussian_filter(background, sigma=15)
        background = background / xp.max(background)

    with timeit("generate two views via blending"):
        image = image_gt + background_stength*background

    if length_z_factor != 1:
        with timeit("downscale along z"):
            image_gt = sp.ndimage.zoom(image_gt, zoom=(1/length_z_factor, 1, 1), order=0)
            background = sp.ndimage.zoom(background, zoom=(1/length_z_factor, 1, 1), order=0)
            image = sp.ndimage.zoom(image, zoom=(1/length_z_factor, 1, 1), order=0)

    if add_noise:
        with timeit("add noise"):
            image = backend.to_numpy(image)
            image = random_noise(image, mode='speckle', var=0.5)
            image = backend.to_backend(image)

    with timeit("scale image intensities"):
        zero_level = xp.random.uniform(95, 10, size=image_gt.shape)
        image = zero_level+300*image

    return image_gt.astype('f4', copy=False), \
           background.astype('f4', copy=False), \
           image.astype('f4', copy=False),