import math

import numpy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def classic_deskew(
    image: xpArray,
    depth_axis: int,
    lateral_axis: int,
    dx: float,
    dz: float,
    angle: float,
    camera_orientation: int = 0,
    flip_depth_axis: bool = False,
    epsilon: float = 1e-2,
    order: int = 1,
    padding: bool = True,
    copy: bool = True,
    internal_dtype=None,
):
    """Classic Deskewing.

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    depth_axis  : Depth axis.
    lateral_axis  : Lateral axis.
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis in degrees
    camera_orientation : Camera orientation correction expressed as a number of 90 deg rotations to be performed
        per 2D image in stack -- if required.
    flip_depth_axis : Flips depth axis to deskew in the opposite orientation (True for view 0 and False for view 1)
    epsilon : relative tolerance to non-integral shifts: if the shift is not _exactly_ integer, to which extent
        do we actually tolerate that when switching between the 'roll' method and full interpolation approach.
    order : interpolation order (when necessary only!)
    padding : Apply padding, or not.
    copy : Set to True to force copy of images.
    internal_dtype : internal dtype to perform computation

    Returns
    -------
    Deskewed image

    """

    # computes the dimension-less parameters:
    shift = dx * math.sin(angle * math.pi / 180) / dz
    lateral_scaling = math.cos(angle * math.pi / 180)

    image = classic_deskew_dimensionless(
        image=image,
        shift=shift,
        depth_axis=depth_axis,
        lateral_axis=lateral_axis,
        flip_depth_axis=flip_depth_axis,
        lateral_scaling=lateral_scaling,
        camera_orientation=camera_orientation,
        epsilon=epsilon,
        order=order,
        padding=padding,
        copy=copy,
        internal_dtype=internal_dtype,
    )

    return image


def classic_deskew_dimensionless(
    image: xpArray,
    depth_axis: int,
    lateral_axis: int,
    shift: float,
    lateral_scaling: float = 1.0,
    flip_depth_axis: bool = False,
    camera_orientation: int = 0,
    epsilon: float = 1e-2,
    order: int = 1,
    padding: bool = False,
    copy: bool = True,
    internal_dtype=None,
):
    """Classic Deskewing with the dimensionless parametrisation.

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    depth_axis  : Depth axis.
    lateral_axis  : Lateral axis.
    shift  : Shift to apply along lateral axis per plane stacked along the depth axis.
    lateral_scaling : Scaling necessary along the lateral axis to preserve voxel dimensions.
    flip_depth_axis : Flips depth axis to deskew in the opposite orientation
        (True for view 0 and False for view 1)
    camera_orientation : Camera orientation correction expressed as a number of 90 deg rotations
        to be performed per 2D image in stack -- if required.
    epsilon : relative tolerance to non-integral shifts: if the shift is not _exactly_ integer, to which
        extent do we actually tolerate that when switching between the 'roll' method and full interpolation approach.
    order : interpolation order (when necessary only!)
    padding : Apply padding, or not.
    copy : Set to True to force copy of images.
    internal_dtype : internal dtype to perform computation

    Returns
    -------
    Deskewed image

    """

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    if internal_dtype is None:
        internal_dtype = xp.float16

    if type(Backend.current()) is NumpyBackend and internal_dtype == xp.float16:
        internal_dtype = xp.float32

    # Save original dtype:
    original_dtype = image.dtype

    # Move images to backend.
    image = Backend.to_backend(image, dtype=internal_dtype, force_copy=copy)

    # First we compute the permutation that will reorder the axis so that the depth
    # and lateral axis are the first axis in the image:
    permutation = (depth_axis, lateral_axis) + tuple(
        axis for axis in range(image.ndim) if axis not in [depth_axis, lateral_axis]
    )
    permutation = numpy.asarray(permutation)
    inverse_permutation = numpy.argsort(permutation)

    # We apply the permutation:
    image = xp.transpose(image, axes=permutation)

    # Camera orientation correction, if required:
    image = xp.rot90(image, k=camera_orientation, axes=(1, 2))

    if flip_depth_axis:
        image = xp.flip(image, axis=0)

    # Image height and depth:
    height = image.shape[1]

    # Padding:
    if padding:
        pad_width_z = int(round(abs(shift) * height / 2))
        pad_width = (
            (pad_width_z, pad_width_z),
            (0, 0),
            (0, 0),
        )
        image = xp.pad(image, pad_width=pad_width)

    # Is the shift integral?
    integral_shift = abs(shift - int(shift)) / shift < epsilon

    if integral_shift:
        # Shift is integral:
        for hi in range(height):
            image[:, hi, ...] = xp.roll(image[:, hi, ...], shift=shift * hi, axis=0)
    else:
        # Shift is not integral, we go full interpolation
        # Deskew matrix:
        matrix = xp.asarray([[1, -shift, 0], [0, 1, 0], [0, 0, 1]])

        # deskew:
        image = sp.ndimage.affine_transform(image, matrix, order=order)

    # rescale the data along lateral axis to preserve voxel dimensions:
    # if lateral_scaling != 1.0:
    #     zoom = [1, ] * 3
    #     zoom[lateral_axis] = lateral_scaling
    #     image = sp.ndimage.zoom(image, zoom=zoom, order=order)

    # flip along axis z
    if flip_depth_axis:
        image = xp.flip(image, axis=0)

    # We apply the inverse permutation:
    image = xp.transpose(image, axes=inverse_permutation)

    # Cast back to original dtype, if needed:
    image = image.astype(original_dtype, copy=False)

    return image
