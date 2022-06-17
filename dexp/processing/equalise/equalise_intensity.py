import warnings

import numpy
from arbol import aprint

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def equalise_intensity(
    image1: xpArray,
    image2: xpArray,
    zero_level=90,
    quantile_low=0.01,
    quantile_high=0.99,
    project_axis: int = 0,
    max_voxels: int = 1e7,
    correction_ratio: float = None,
    copy: bool = True,
    internal_dtype=None,
):
    """
    Equalise intensity between two images

    Parameters
    ----------
    image1 : first image to equalise
    image2 : second image to equalise
    zero_level : zero level -- removes this value if that's the minimal voxel value expected for both images
    quantile_low, quantile_high : quantile for computing the robust min and max values in image
    project_axis : Axis over which to project image to speed up computation
    max_voxels : maximal number of voxels to use to compute min and max values.
    correction_ratio : If provided, this value is used instead of calculating the equalisation value based
        on the two images. This is the value that should be multiplied to the second image (image2)
        so as to equalise it to teh first image (image1).
    copy : Set to True to force copy of images.
    internal_dtype : dtype to use internally for computation.

    Returns
    -------
    The two arrays intensity equalised.

    """
    xp = Backend.get_xp_module()

    # Sanity checks:
    if image1.dtype != image2.dtype:
        raise ValueError("Two images must have same dtype!")

    if internal_dtype is None:
        internal_dtype = image1.dtype

    if type(Backend.current()) is NumpyBackend:
        internal_dtype = numpy.float32

    # Save original dtype:
    original_dtype = image1.dtype

    # Move images to backend.
    image1 = Backend.to_backend(image1, dtype=internal_dtype, force_copy=copy)
    image2 = Backend.to_backend(image2, dtype=internal_dtype, force_copy=copy)

    # If not provided, compute correction ratio:
    if correction_ratio is None:

        # To reduce computation effort, we operate on the max projected images:
        proj_image1 = xp.max(image1, axis=project_axis)
        proj_image2 = xp.max(image2, axis=project_axis)

        # instead of computing the rations on all voxels, we reduce to a dense subset:
        reduction = max(1, 4 * (int(proj_image1.size / max_voxels) // 4))
        strided_image1 = proj_image1.ravel()[::reduction].astype(numpy.float32, copy=False)
        strided_image2 = proj_image2.ravel()[::reduction].astype(numpy.float32, copy=False)

        # We determine a robust maximum for voxel intensities:
        highvalue1 = xp.percentile(strided_image1, q=quantile_high * 100, method="higher")
        highvalue2 = xp.percentile(strided_image2, q=quantile_high * 100, method="higher")

        # we set a 'high-value' threshold to the minimum of both robust maximums:
        threshold = min(highvalue1, highvalue2)

        # We find the voxels that are above that threshold in both images:
        mask1 = strided_image1 >= threshold
        mask2 = strided_image2 >= threshold
        mask = xp.logical_and(mask1, mask2)

        # And extract these values from both images:
        highvalues1 = strided_image1[mask]
        highvalues2 = strided_image2[mask]

        # And a robust minimum:
        lowvalue1 = xp.percentile(strided_image1, q=quantile_low * 100)
        lowvalue2 = xp.percentile(strided_image2, q=quantile_low * 100)

        # Compute ratios:
        range1 = xp.clip(highvalues1 - lowvalue1, 0, None)
        range2 = xp.clip(highvalues2 - lowvalue2, 0, None)
        ratios = range1 / range2

        # Keep only valid ratios:
        valid = xp.logical_and(range1 != 0, range2 != 0)
        ratios = ratios[valid]

        # Free memory:
        del mask, mask1, mask2, highvalues1, highvalues2, strided_image1, strided_image2

        # Number of values in ratio:
        nb_values = ratios.size
        if nb_values < 16:
            warnings.warn(
                f"No enough values ({nb_values}<16) to compute correction ratio! Relax percentile or reduction! "
            )
            image1 = image1.astype(original_dtype, copy=False)
            image2 = image2.astype(original_dtype, copy=False)
            return image1, image2, 1.0
        elif nb_values < 128:
            aprint(
                f"Warning: too few ratio values ({nb_values}<128)"
                + "to compute correction ratio! Relax percentile or reduction!"
            )

        correction_ratio = xp.median(ratios.astype(internal_dtype, copy=False))
    else:
        correction_ratio = Backend.to_backend(correction_ratio, dtype=internal_dtype)

    # remove zero level and clip:
    if zero_level != 0:
        # FIXME: shouldn't the order be inverted? this might create edges
        image1 = xp.clip(image1, a_min=zero_level, a_max=None, out=image1)
        image1 -= zero_level
        image2 = xp.clip(image2, a_min=zero_level, a_max=None, out=image2)
        image2 -= zero_level

    # compute inverse ratios and cast to internal type:
    inverse_correction_ratio = 1 / correction_ratio
    correction_ratio = correction_ratio.astype(dtype=internal_dtype)
    inverse_correction_ratio = inverse_correction_ratio.astype(dtype=internal_dtype)

    # Apply correction ratio while only increasing intensity of the dimmer image:
    if correction_ratio > 1:
        image2 *= correction_ratio
    else:
        image1 *= inverse_correction_ratio

    # cast back to original dtype:
    image1 = image1.astype(original_dtype, copy=False)
    image2 = image2.astype(original_dtype, copy=False)

    # KEEP FOR DEBUGGING:
    # from napari import Viewer
    # import napari
    # with napari.gui_qt():
    #     def _c(array):
    #         return backend.to_numpy(array)
    #
    #     viewer = Viewer()
    #     viewer.add_image(_c(image1), name='image_1')
    #     viewer.add_image(_c(image2), name='image_2')

    return image1, image2, correction_ratio
