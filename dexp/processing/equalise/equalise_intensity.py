import numpy

from dexp.processing.backends.backend import Backend


def equalise_intensity(backend: Backend, image1, image2, zero_level=90, percentile=0.99, max_voxels=1e6, dtype=None):
    if image1.dtype != image2.dtype:
        raise ValueError("Two images must have same dtype!")

    image1 = backend.to_backend(image1)
    image2 = backend.to_backend(image2)

    if dtype is None:
        dtype = image1.dtype

    xp = backend.get_xp_module()

    reduction = max(1, 4 * (int(image1.size / max_voxels) // 4))

    strided_image1 = image1.ravel()[::reduction].astype(numpy.float32, copy=False)
    strided_image2 = image2.ravel()[::reduction].astype(numpy.float32, copy=False)

    highvalue1 = xp.percentile(strided_image1, q=percentile * 100)
    highvalue2 = xp.percentile(strided_image2, q=percentile * 100)

    lowvalue1 = xp.percentile(strided_image1, q=(1 - percentile) * 100)
    lowvalue2 = xp.percentile(strided_image2, q=(1 - percentile) * 100)

    mask1 = strided_image1 >= highvalue1
    mask2 = strided_image2 >= highvalue2

    mask = mask1 * mask2

    highvalues1 = strided_image1[mask]
    highvalues2 = strided_image2[mask]

    ratios = (highvalues1 - lowvalue1) / (highvalues2 - lowvalue2)
    del mask, mask1, mask2, highvalues1, highvalues2, strided_image1, strided_image2

    nb_values = ratios.size
    if nb_values < 128:
        raise ValueError("Too few ratio values to compute correction ratio! Relax percentile or reduction! ")

    correction_ratio = xp.percentile(ratios.astype(numpy.float32, copy=False), q=50)
    inverse_correction_ratio = 1 / correction_ratio

    if image1.itemsize >= 4:
        image1 = image1.astype(xp.float32)
        image2 = image2.astype(xp.float32)
    elif image1.itemsize <= 2:
        image1 = image1.astype(xp.float16)
        image2 = image2.astype(xp.float16)

    if zero_level != 0:
        image1 -= zero_level
        image2 -= zero_level

    image1.clip(0, None, out=image1)
    image2.clip(0, None, out=image2)

    correction_ratio = correction_ratio.astype(dtype=image1.dtype)
    inverse_correction_ratio = inverse_correction_ratio.astype(dtype=image1.dtype)

    if correction_ratio > 1:
        image2 *= correction_ratio
    else:
        image1 *= inverse_correction_ratio

    image1 = image1.astype(dtype, copy=False)
    image2 = image2.astype(dtype, copy=False)

    return image1, image2, correction_ratio
