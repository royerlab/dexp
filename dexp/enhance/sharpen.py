import numexpr
from math import sqrt

import numpy
import scipy
from gputools import OCLArray, gaussian_filter
from gputools.convolve import median_filter
from gputools.denoise import nlm3


def fast_median_filter(image, size):
    filter_size = 3
    nb_steps = size//filter_size
    #print(f'nb_steps={nb_steps}, filter_size={filter_size}')
    temp1 = OCLArray.from_array(image)
    temp2 = OCLArray.empty_like(numpy.zeros(image.shape, image.dtype))
    for step in range(nb_steps):
        #print(f'step={step+1}')
        median_filter(temp1,  size=filter_size, res_g=temp2)
        temp1, temp2 = temp2, temp1
    if nb_steps % 2 == 0:
        return temp1.get()
    else:
        return temp2.get()

def fast_hybrid_filter(image, size, sigma=1):

    image = image.astype(numpy.float32, copy=False)

    filter_size = 3
    nb_steps = (size//filter_size)//2

    #print(f'nb_steps={nb_steps}, filter_size={filter_size}, sigma={sigma}')
    temp1 = OCLArray.from_array(image)
    temp2 = OCLArray.empty_like(numpy.zeros(image.shape, image.dtype))
    for step in range(nb_steps):
        #print(f'step={step+1}')
        median_filter(temp1,   size=filter_size, res_g=temp2)
        gaussian_filter(temp2, sigma=sigma, res_g=temp1)

    return temp1.get()

def median_sharpening(image, size=20):

    median_filtered = fast_median_filter(image, size)
    #median_filtered =  median_filter(image, size=size)

    median_sharpened = numpy.clip(image.astype(numpy.float16) - median_filtered, 0, numpy.math.inf)

    return median_sharpened, median_filtered


def gaussian_sharpening(image, size=20):

    # Three sigmas is the support of the filter
    gaussian_filtered = gaussian_filter(image, sigma=size//3)

    gaussian_sharpened = numpy.clip(image - gaussian_filtered, 0, numpy.math.inf)

    return gaussian_sharpened, gaussian_filtered


def hybrid_sharpening(image, size=20, sigma=2):

    hybrid_filtered = fast_hybrid_filter(image, size, sigma)

    hybrid_sharpened = numpy.clip(image.astype(numpy.float16) - hybrid_filtered, 0, numpy.math.inf)

    return hybrid_sharpened, hybrid_filtered


def sharpen(image,
            mode='hybrid',
            size=20,
            normalise_range=True, min=None, max=None, epsilon=0.1,
            clear_margins=True, margin_pad=True, margin=8,
            denoise=False
            ):

    #image = image.astype(numpy.float32, copy=False)

    if mode == 'gaussian':
        sharpened,_ = gaussian_sharpening(image, size)

    if mode == 'median':
        sharpened,_ = median_sharpening(image, size)

    if mode == 'hybrid':
        sharpened,_ = hybrid_sharpening(image, size)

    if normalise_range:
        if not min is None:
            org_min = min
        else:
            org_min = numpy.percentile(image, epsilon)

        if not max is None:
            org_max = max
        else:
            org_max = numpy.percentile(image, 100 - epsilon)


        new_min = numpy.percentile(sharpened, epsilon)
        new_max = numpy.percentile(sharpened, 100-epsilon)

        scaling = (org_max-org_min)/(new_max-new_min)
        offset  = org_min-scaling*new_min

        sharpened = numexpr.evaluate("sharpened*scaling + offset")

    if clear_margins:
        sharpened = sharpened[margin:-margin, margin:-margin, margin:-margin]
        if margin_pad:
            sharpened = numpy.pad(sharpened, pad_width=margin, mode='constant')

    if denoise:
        sharpened = sharpened.astype(numpy.uint16, copy=False)
        sharpened = median_filter(sharpened, size=3)

    return sharpened




