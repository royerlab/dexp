import numpy
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise


def generate_fusion_test_data(length=128):
    image_gt = binary_blobs(length=length, n_dim=3, blob_size_fraction=0.07, volume_fraction=0.01).astype('f4')
    image_gt = gaussian(image_gt, sigma=1)
    image_gt = image_gt / numpy.max(image_gt)

    image_highq = image_gt.copy()
    image_lowq = image_gt.copy()
    image_lowq = gaussian(image_lowq, sigma=7)

    blend = binary_blobs(length=128, n_dim=3, blob_size_fraction=0.3, volume_fraction=0.5).astype('f4')
    blend = gaussian(blend, sigma=2)
    blend = blend / numpy.max(blend)

    image1 = image_highq * blend + image_lowq * (1 - blend)
    image2 = image_highq * (1 - blend) + image_lowq * blend

    image1 = 95+300*random_noise(image1, mode='speckle', var=0.5)
    image2 = 95+300*random_noise(image2, mode='speckle', var=0.5)

    return image_gt, image_lowq, blend, image1, image2