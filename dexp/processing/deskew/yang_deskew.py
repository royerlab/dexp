import math

import numpy

from dexp.utils import xpArray
from dexp.utils.backends import Backend, NumpyBackend


def yang_deskew(
    image: xpArray,
    depth_axis: int,
    lateral_axis: int,
    flip_depth_axis: bool,
    dx: float,
    dz: float,
    angle: float,
    camera_orientation: int = 0,
    num_split: int = 4,
    internal_dtype=None,
    padding: None = None,
):
    """'Yang' Deskewing as done in Yang et al. 2019 ( https://www.biorxiv.org/content/10.1101/2020.09.22.309229v2 )

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    depth_axis  : Depth axis.
    lateral_axis  : Lateral axis.
    flip_depth_axis   : Flips image to deskew in the opposite orientation (True for view 0 and False for view 1)
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis in degrees
    camera_orientation : Camera orientation correction expressed
        as a number of 90 deg rotations to be performed per 2D image in stack.
    num_split : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU
    internal_dtype : internal dtype to perform computation
    padding : dummy argument for compatibility

    Returns
    -------
    Deskewed image

    """

    # compute dimensionless parameters:
    xres = dx * math.cos(angle * math.pi / 180)
    resample_factor = int(round(dz / xres))
    lateral_scaling = xres / dx

    image = yang_deskew_dimensionless(
        image=image,
        depth_axis=depth_axis,
        lateral_axis=lateral_axis,
        flip_depth_axis=flip_depth_axis,
        resample_factor=resample_factor,
        lateral_scaling=lateral_scaling,
        camera_orientation=camera_orientation,
        num_split=num_split,
        internal_dtype=internal_dtype,
    )

    return image


def yang_deskew_dimensionless(
    image: xpArray,
    depth_axis: int,
    lateral_axis: int,
    flip_depth_axis: bool,
    resample_factor: int,
    lateral_scaling: float,
    camera_orientation: int = 0,
    num_split: int = 4,
    internal_dtype=None,
):
    """'Yang' Deskewing as done in Yang et al. 2019 ( https://www.biorxiv.org/content/10.1101/2020.09.22.309229v2 )

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    depth_axis  : Depth axis.
    lateral_axis  : Lateral axis.
    flip_depth_axis   : Flips image to deskew in the opposite orientation (True for view 0 and False for view 1)
    resample_factor : Resampling factor
    lateral_scaling : Lateral scaling
    camera_orientation : Camera orientation correction expressed as a
        number of 90 deg rotations to be performed per 2D image in stack.
    num_split : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU
    internal_dtype : internal dtype to perform computation

    Returns
    -------
    Deskewed image

    """
    # we don't want to move the image to the backend just now,
    # as it might be a very large image and we can actually defer moving it to the backend as
    # after splitting...
    xp = Backend.get_xp_module(image)

    # We save the original dtype:
    original_dtype = image.dtype

    # Default internal dtype is the same as the input image:
    if internal_dtype is None:
        internal_dtype = image.dtype

    # Numpy backend does not support float16:
    if type(Backend.current()) is NumpyBackend and internal_dtype == xp.float16:
        internal_dtype = xp.float32

    # First we compute the permutation that will reorder the axis so that the depth and
    # lateral axis are the first axis in the image:
    permutation = (depth_axis, lateral_axis) + tuple(
        axis for axis in range(image.ndim) if axis not in [depth_axis, lateral_axis]
    )
    permutation = numpy.asarray(permutation)
    inverse_permutation = numpy.argsort(permutation)

    # We apply the permutation:
    image = xp.transpose(image, axes=permutation)

    if flip_depth_axis:
        image = xp.flip(image, axis=0)

    # rotate the data
    # Note: the weird 1+co mod 4 is due to the fact that Bin's original
    # code was implemented for a different orientation...
    image = xp.rot90(image, k=(1 + camera_orientation) % 4, axes=(1, 2))

    # deskew and rotate
    image = _resampling_vertical_split(
        image,
        resample_factor=resample_factor,
        lateral_scaling=lateral_scaling,
        num_split=num_split,
        internal_dtype=internal_dtype,
    )

    # flip along axis x
    if flip_depth_axis:
        xp = Backend.get_xp_module(image)
        # _resampling_vertical_split transposes axis 0 with axis 2, so we flip along 2:
        image = xp.flip(image, axis=2)

    # We apply the inverse permutation:
    image = xp.transpose(image, axes=inverse_permutation)

    # Cast back to original dtype:
    image = image.astype(original_dtype, copy=False)

    return image


def _resampling_vertical_split(
    image, resample_factor: int, lateral_scaling: float, num_split: int = 4, internal_dtype=None
):
    """Same as resampling_vertical but splits the input image so that computation can fit in GPU memory.

    Parameters
    ----------
    image   : input image (skewed 3D stack)
    resample_factor : Resampling factor
    lateral_scaling : Lateral scaling
    internal_dtype : internal dtype to perform computation

    Returns
    -------
    Resampled image.
    Important note: for scalability reasons, the returned image is always numpy image.

    """

    if num_split == 1:
        output = _resampling_vertical(
            image, resample_factor=resample_factor, lateral_scaling=lateral_scaling, internal_dtype=internal_dtype
        )
    else:
        xp = Backend.get_xp_module(image)
        data_gpu_splits = xp.array_split(image, num_split, axis=1)
        data_cpu_splits = []
        for k in range(num_split):
            data_resampled = _resampling_vertical(
                data_gpu_splits[k],
                resample_factor=resample_factor,
                lateral_scaling=lateral_scaling,
                internal_dtype=internal_dtype,
            )
            data_cpu_splits.append(Backend.to_numpy(data_resampled, dtype=image.dtype))

        output = numpy.concatenate(data_cpu_splits, axis=1)

    return output


def _resampling_vertical(image, resample_factor: int, lateral_scaling: float, internal_dtype=None):
    """Resampling of the image by interpolation along vertical direction.
    Here we assume the dz is integer multiple of dx * cos(angle * pi / 180),
    one can also pre-interpolate the data within along the z' axis if this is not the case

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    resample_factor :  Resampling factor
    lateral_scaling : Lateral scaling
    internal_dtype : internal dtype to perform computation

    Returns
    -------
    Resampled image


    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    # Move images to backend.
    image = Backend.to_backend(image, dtype=internal_dtype)

    (nz, ny, nx) = image.shape
    dtype = image.dtype

    nz_new, ny_new, nx_new = len(range(0, nx, resample_factor)), ny, nx + nz * resample_factor
    data_reassign = xp.zeros((nz_new, ny_new, nx_new), internal_dtype)

    for x in range(nx):
        x_start = x
        x_end = nz * resample_factor + x
        data_reassign[x // resample_factor, :, x_start:x_end:resample_factor] = image[:, :, x].T
    del image

    # rescale the data, interpolate along z
    data_rescale = sp.ndimage.zoom(data_reassign, zoom=(resample_factor, 1, 1), order=1)
    del data_reassign

    data_interp = xp.zeros((nz_new, ny_new, nx_new), dtype)

    for z in range(nz_new):
        for k in range(resample_factor):
            data_interp[z, :, k::resample_factor] = data_rescale[z * resample_factor - k, :, k::resample_factor]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    image_final = sp.ndimage.zoom(data_interp[1:], zoom=(1, 1, lateral_scaling), order=1)

    return image_final
