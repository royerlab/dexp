import numpy

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.numpy_backend import NumpyBackend


def equalise_intensity(backend: Backend,
                       image1,
                       image2,
                       zero_level=90,
                       quantile=0.99,
                       max_voxels=1e6,
                       copy: bool = True,
                       internal_dtype=numpy.float16):
    """
    Equalise intensity between two images

    Parameters
    ----------
    backend : backend to use
    image1 : first image to equalise
    image2 : second image to equalise
    zero_level : zero level -- removes this value if that's the minimal voxel value expected for both images
    quantile : quantile for computinmg the robust min and max values in image
    max_voxels : maximal number of voxels to use to compute min and max values.
    copy : Set to True to force copy of images.
    internal_dtype : dtype to use internally for computation.

    Returns
    -------
    The two arrays intensity equalised.

    """
    if image1.dtype != image2.dtype:
        raise ValueError("Two images must have same dtype!")

    if type(backend) is NumpyBackend:
        internal_dtype = numpy.float32

    original_dtype = image1.dtype
    image1 = backend.to_backend(image1, dtype=internal_dtype, force_copy=copy)
    image2 = backend.to_backend(image2, dtype=internal_dtype, force_copy=copy)

    xp = backend.get_xp_module()

    reduction = max(1, 4 * (int(image1.size / max_voxels) // 4))

    strided_image1 = image1.ravel()[::reduction].astype(numpy.float32, copy=False)
    strided_image2 = image2.ravel()[::reduction].astype(numpy.float32, copy=False)

    highvalue1 = xp.percentile(strided_image1, q=quantile * 100)
    highvalue2 = xp.percentile(strided_image2, q=quantile * 100)

    lowvalue1 = xp.percentile(strided_image1, q=(1 - quantile) * 100)
    lowvalue2 = xp.percentile(strided_image2, q=(1 - quantile) * 100)

    mask1 = strided_image1 >= highvalue1
    mask2 = strided_image2 >= highvalue2

    mask = xp.logical_and(mask1, mask2)

    highvalues1 = strided_image1[mask]
    highvalues2 = strided_image2[mask]

    # compute ratios:
    range1 = highvalues1 - lowvalue1
    range2 = highvalues2 - lowvalue2
    ratios = (range1 / range2)

    # keep only valid ratios:
    valid = xp.logical_and(range1!=0, range2!=0)
    ratios = ratios[valid]

    # Free memory:
    del mask, mask1, mask2, highvalues1, highvalues2, strided_image1, strided_image2

    nb_values = ratios.size
    if nb_values < 128:
        raise ValueError(f"Too few ratio values ({nb_values}) to compute correction ratio! Relax percentile or reduction! ")

    correction_ratio = xp.percentile(ratios.astype(internal_dtype, copy=False), q=50)
    inverse_correction_ratio = 1 / correction_ratio

    if zero_level != 0:
        image1 -= zero_level
        image2 -= zero_level

    image1.clip(0, None, out=image1)
    image2.clip(0, None, out=image2)

    correction_ratio = correction_ratio.astype(dtype=internal_dtype)
    inverse_correction_ratio = inverse_correction_ratio.astype(dtype=internal_dtype)

    if correction_ratio > 1:
        image2 *= correction_ratio
    else:
        image1 *= inverse_correction_ratio

    image1 = image1.astype(original_dtype, copy=False)
    image2 = image2.astype(original_dtype, copy=False)

    # from napari import Viewer
    # import napari
    # with napari.gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image1), name='image_1')
    #     viewer.add_image(_c(image1), name='image_2')

    return image1, image2, correction_ratio
