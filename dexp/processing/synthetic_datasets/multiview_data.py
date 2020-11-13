from typing import Tuple, Optional

import numpy
from skimage.data import binary_blobs
from skimage.util import random_noise

from dexp.processing.backends.backend import Backend
from dexp.utils.timeit import timeit


def generate_fusion_test_data(backend: Backend,
                              length_xy: Optional[int] = 320,
                              length_z_factor: Optional[int] = 4,
                              add_noise: Optional[bool] = True,
                              shift: Optional[Tuple[int, ...]] = None,
                              volume_fraction: Optional[float] = 0.8,
                              amount_low: Optional[float] = 1,
                              zero_level: Optional[float] = 95,
                              odd_dimension: Optional[float] = True,
                              z_overlap: Optional[float] = None):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    with timeit("generate blob images"):
        image_gt = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')
        blend_a = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.2, volume_fraction=volume_fraction).astype('f4')
        blend_b = binary_blobs(length=length_xy, n_dim=3, blob_size_fraction=0.2, volume_fraction=volume_fraction).astype('f4')

    with timeit("convert blob images to backend"):
        image_gt = backend.to_backend(image_gt)
        blend_a = backend.to_backend(blend_a)
        blend_b = backend.to_backend(blend_b)

    with timeit("prepare high/low image pair"):
        image_gt = sp.ndimage.gaussian_filter(image_gt, sigma=1)
        image_gt = image_gt / xp.max(image_gt)
        image_highq = image_gt.copy()
        image_lowq = image_gt.copy()
        image_lowq = amount_low * sp.ndimage.gaussian_filter(image_lowq, sigma=7)

    with timeit("prepare blend maps"):
        blend_a = sp.ndimage.gaussian_filter(blend_a, sigma=2)
        blend_a = blend_a / numpy.max(blend_a)
        blend_b = sp.ndimage.gaussian_filter(blend_b, sigma=2)
        blend_b = blend_b / numpy.max(blend_b)

    if z_overlap is not None:
        with timeit("add z bias to blend maps"):
            length = blend_a.shape[0]
            bias_vector = xp.linspace(0, 1, num=length)
            new_shape = tuple(s if i == 0 else 1 for i, s in enumerate(blend_a.shape))
            bias_vector = xp.reshape(bias_vector, newshape=new_shape)
            blend_a *= (1 - bias_vector) ** z_overlap
            blend_b *= (bias_vector) ** z_overlap

    with timeit("generate two views via blending"):
        image1 = image_highq * blend_a + image_lowq * blend_b
        image2 = image_highq * blend_b + image_lowq * blend_a

    if length_z_factor != 1:
        with timeit("downscale along z"):
            image_gt = sp.ndimage.zoom(image_gt, zoom=(1 / length_z_factor, 1, 1), order=0)
            image_lowq = sp.ndimage.zoom(image_lowq, zoom=(1 / length_z_factor, 1, 1), order=0)
            blend_a = sp.ndimage.zoom(blend_a, zoom=(1 / length_z_factor, 1, 1), order=0)
            blend_b = sp.ndimage.zoom(blend_b, zoom=(1 / length_z_factor, 1, 1), order=0)
            image1 = sp.ndimage.zoom(image1, zoom=(1 / length_z_factor, 1, 1), order=0)
            image2 = sp.ndimage.zoom(image2, zoom=(1 / length_z_factor, 1, 1), order=0)

    if odd_dimension:
        image_gt = xp.pad(image_gt, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')
        image_lowq = xp.pad(image_lowq, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')
        blend_a = xp.pad(blend_a, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')
        blend_b = xp.pad(blend_b, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')
        image1 = xp.pad(image1, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')
        image2 = xp.pad(image2, pad_width=((0, 1), (0, 0), (1, 0)), mode='edge')

    if add_noise:
        with timeit("add noise"):
            image1 = backend.to_numpy(image1)
            image2 = backend.to_numpy(image2)
            image1 = random_noise(image1, mode='speckle', var=0.5)
            image2 = random_noise(image2, mode='speckle', var=0.5)
            image1 = backend.to_backend(image1)
            image2 = backend.to_backend(image2)

    if shift is not None:
        with timeit("Shift second view relative to first"):
            image2 = sp.ndimage.shift(image2, shift=shift, order=1)

    with timeit("scale image intensities"):
        image1 = zero_level + 300 * image1
        image2 = zero_level + 300 * image2
        image_gt = zero_level + 300 * image_gt
        image_lowq = zero_level + 300 * image_lowq

    return image_gt.astype('f4', copy=False), \
           image_lowq.astype('f4', copy=False), \
           blend_a.astype('f4', copy=False), \
           blend_b.astype('f4', copy=False), \
           image1.astype('f4', copy=False), \
           image2.astype('f4', copy=False)