import numpy
import scipy
from scipy.ndimage import gaussian_filter
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise


def generate_fusion_test_data(length=128, add_noise=True):
    image_gt = binary_blobs(length=length, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.1).astype('f4')
    image_gt = gaussian(image_gt, sigma=1)
    image_gt = image_gt / numpy.max(image_gt)

    image_highq = image_gt.copy()
    image_lowq = image_gt.copy()
    image_lowq = gaussian(image_lowq, sigma=7)

    blend_a = binary_blobs(length=128, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.8).astype('f4')
    blend_a = gaussian(blend_a, sigma=2)
    blend_a = blend_a / numpy.max(blend_a)
    blend_b = binary_blobs(length=128, n_dim=3, blob_size_fraction=0.2, volume_fraction=0.8).astype('f4')
    blend_b = gaussian(blend_b, sigma=2)
    blend_b = blend_b / numpy.max(blend_b)

    # blend_a = blend_a
    # blend_b = 1-blend_b

    image1 = image_highq * blend_a + image_lowq * blend_b
    image2 = image_highq * blend_b + image_lowq * blend_a

    if add_noise:
        image1 = random_noise(image1, mode='speckle', var=0.5)
        image2 = random_noise(image2, mode='speckle', var=0.5)

    image1 = 95+300*image1
    image2 = 95+300*image2
    image_gt = 95+300*image_gt
    image_lowq= 95+300*image_lowq

    return image_gt, image_lowq, blend_a, blend_b, image1, image2