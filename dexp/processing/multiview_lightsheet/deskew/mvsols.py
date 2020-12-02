import numpy

from dexp.processing.backends.backend import Backend


def process_view(image,
                 flip:bool,
                 dx: float,
                 dz: float,
                 angle: float,
                 num_split: int):

    xp = Backend.get_xp_module()

    # move to backend:
    image = Backend.to_backend(image)
    if flip:
        image = xp.flip(image, axis=0)

    # rotate the data
    image = xp.rot90(image, k=1, axes=(1, 2))
    image = xp.array(image, copy=True, order='C')

    # deskew and rotate
    image = resampling_vertical_cupy_split(image,
                                                   dx=dx,
                                                   dz=dz,
                                                   angle=angle,
                                                   num_split=num_split)

    # flip along axis x
    if flip:
        image = xp.flip(image, axis=2)

    return image


def resampling_vertical_cupy_split(image,
                                   dz: float = 1.0,
                                   dx: float = 0.2,
                                   angle: float = 45,
                                   num_split: int = 4):

    xp = Backend.get_xp_module()

    data_gpu_splits = xp.array_split(image, num_split, axis=1)
    for k in range(num_split):
        data_resampled = resampling_vertical_cupy(data_gpu_splits[k], dz, dx, angle=angle)
        if k == 0:
            output = Backend.to_numpy(data_resampled)
        else:
            output = numpy.concatenate((output, Backend.to_numpy(data_resampled)), axis=1)

    return output


def resampling_vertical_cupy(image,
                             dz: float = 1.0,
                             dx: float = 0.2,
                             angle: float = 45):
    """resampling of the data by interpolation along vertical direction.
    Here we assume the dz is integer multiple of dx * cos(angle * pi / 180),
    one can also pre interpolate the data within along the z' axis if this is not the case
     :param
     data   : ndarray in cupy, 3D stack
     dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
     dx     : float, pixel size of the camera
     angle  : float, incident angle of the light shee, angle between the light sheet and the optical axis
     num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU"""

    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    (nz, ny, nx) = image.shape
    dtype = image.dtype

    zres = dz * xp.sin(angle * xp.pi / 180)
    xres = dx * xp.cos(angle * xp.pi / 180)

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
    data_rescale = sp.ndimage.zoom(data_reassign, (resample_factor_int, 1, 1))
    del data_reassign

    data_interp = xp.zeros((nz_new, ny_new, nx_new), dtype)

    for z in range(nz_new):
        for k in range(resample_factor_int):
            data_interp[z, :, k::resample_factor_int] = \
                data_rescale[z * resample_factor_int - k, :, k::resample_factor_int]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    data_final = sp.ndimage.zoom(data_interp[1:], (1, 1, xres / dx))

    return data_final