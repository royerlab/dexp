import numpy

from dexp.processing.backends.backend import Backend


def yang_deskew(image,
                flip: bool,
                dx: float,
                dz: float,
                angle: float,
                num_split: int = 4):
    """  Resampling

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    flip   : True for view 0 and False for view 1
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU

    Returns
    -------

    """
    # we don't want to move the image to the backend just now,
    # as it might be a very large image and we can actually defer moving it to the backend as
    # after splitting...
    xp = Backend.get_xp_module(image)

    if flip:
        image = xp.flip(image, axis=0)

    # rotate the data
    image = xp.rot90(image, k=1, axes=(1, 2))

    # deskew and rotate
    image = resampling_vertical_split(image,
                                      dx=dx,
                                      dz=dz,
                                      angle=angle,
                                      num_split=num_split)

    # flip along axis x
    if flip:
        xp = Backend.get_xp_module(image)
        image = xp.flip(image, axis=2)

    return image


def resampling_vertical_split(image,
                              dz: float = 1.0,
                              dx: float = 0.2,
                              angle: float = 45,
                              num_split: int = 4):
    """ Same as resampling_vertical but splits the input image so that computation can fit in GPU memory.

    Parameters
    ----------
    image   : input image (skewed 3D stack)
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU

    Returns
    -------
    Resampled image.
    Important note: for scalability reasons, the returned image is always numpy image.

    """

    if num_split == 1:
        output = resampling_vertical(image, dz, dx, angle=angle)
    else:
        xp = Backend.get_xp_module(image)
        data_gpu_splits = xp.array_split(image, num_split, axis=1)
        for k in range(num_split):
            data_resampled = resampling_vertical(data_gpu_splits[k], dz, dx, angle=angle)
            if k == 0:
                output = Backend.to_numpy(data_resampled)
            else:
                output = numpy.concatenate((output, Backend.to_numpy(data_resampled)), axis=1)

    return output


def resampling_vertical(image,
                        dz: float,
                        dx: float,
                        angle: float):
    """ Resampling of the image by interpolation along vertical direction.
    Here we assume the dz is integer multiple of dx * cos(angle * pi / 180),
    one can also pre-interpolate the data within along the z' axis if this is not the case

    Parameters
    ----------
    image  : input image (skewed 3D stack)
    dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
    dx     : float, pixel size of the camera
    angle  : float, incident angle of the light sheet, angle between the light sheet and the optical axis
    num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU

    Returns
    -------
    Resampled image


    """
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    (nz, ny, nx) = image.shape
    dtype = image.dtype

    zres = dz * numpy.sin(angle * xp.pi / 180)
    xres = dx * numpy.cos(angle * xp.pi / 180)

    resample_factor = dz / xres
    resample_factor_int = int(round(resample_factor))

    nz_new, ny_new, nx_new = len(range(0, nx, resample_factor_int)), ny, nx + nz * resample_factor_int
    data_reassign = xp.zeros((nz_new, ny_new, nx_new), xp.int16)

    for x in range(nx):
        x_start = x
        x_end = nz * resample_factor_int + x
        data_reassign[x // resample_factor_int, :, x_start:x_end:resample_factor_int] = image[:, :, x].T
    del image

    # rescale the data, interpolate along z
    data_rescale = sp.ndimage.zoom(data_reassign, zoom=(resample_factor_int, 1, 1), order=1)
    del data_reassign

    data_interp = xp.zeros((nz_new, ny_new, nx_new), dtype)

    for z in range(nz_new):
        for k in range(resample_factor_int):
            data_interp[z, :, k::resample_factor_int] = \
                data_rescale[z * resample_factor_int - k, :, k::resample_factor_int]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    image_final = sp.ndimage.zoom(data_interp[1:], zoom=(1, 1, xres / dx), order=1)

    return image_final
