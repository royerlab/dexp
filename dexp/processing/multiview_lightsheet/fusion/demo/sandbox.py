"""The resampling functions used to resample the dual view eSPIM data from x'yz' space to xyz space"""
import os
import time

import cupy as cp
import cupyx.scipy.ndimage as cp_ndimage
import numpy as np
import zarr
from natsort import natsorted
from numcodecs import Blosc


def batch_processing(path, tp_interval=1, num_split=4,
                     compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)):
    """processing all the raw data (in x'yz' space) in path to resampled data (in xyz space)
    if given path is a file, process data in it;
    if given path is a folder, process all the zarr files in it"""

    # determine if the path is a zarr file or a directory (in case that the raw data was acquire in multiple sessions)
    if path.endswith('.zarr'):
        print('input path is a zarr file folder')
        zarr_files = [path]
        main_zarr_path = path.replace('.zarr', '_resampled_intervel_' + str(tp_interval) + '.zarr')
        print('save to:', main_zarr_path)
    elif os.path.isdir(path):
        print('input path is a general folder')
        zarr_files = find_all_files_ends_with(path, '.zarr')
        # create the main zarr folder
        main_zarr_path = os.path.join(path, 'all_tps_resampled_intervel_' + str(tp_interval) + '.zarr')
        print('save to:', main_zarr_path)

    # get the number of time points in each zarr file
    nb_tp_subs = []
    for file in zarr_files:
        root = zarr.open(file, mode='r')
        nb_tp, _, _, _ = root['v0c0']['v0c0'].shape
        nb_tp_subs.append(nb_tp)

    print('number of tp in each file:', nb_tp_subs)
    nb_tp_sub = sum([len(range(0, x, tp_interval)) for x in nb_tp_subs])
    print('number of tp to process:', nb_tp_sub)

    # get attributes
    attrs = _get_attributes(zarr_files[0])
    print(attrs)

    # readout the sub zarr folders, rotate and save (including the MIP)
    for ch in range(attrs['channel']):
        counter = 0
        for n, file in enumerate(zarr_files):
            root = zarr.open(file, mode='r')
            print('arrays in the zarr raw data folder:', root.tree())

            # readout the raw data
            data = readout_zarr_data(root, attrs['view'], ch)

            # process raw data and save to new zarr folder
            for t in range(0, nb_tp_subs[n], tp_interval):
                print("current channel:", ch)
                print("start converting timepoint:", counter * tp_interval)
                start_tp = time.time()
                for v in range(attrs['view']):
                    if v == 0:
                        data_rotate, data_mip = process_v0(data[0], t, attrs, num_split)
                    else:
                        data_rotate, data_mip = process_v1(data[1], t, attrs, num_split)

                    # create the groups in the main_zarr_path folder
                    if n == 0 and t == 0 and v == 0:
                        nz, ny, nx = data_rotate.shape
                        if not os.path.isdir(main_zarr_path):
                            root_all_tp = zarr.open(main_zarr_path, mode='w')
                            create_arrays_in_zarr(root_all_tp, attrs,
                                                  nb_tp_sub, nz, ny, nx, compressor)

                    # get the zarr arrays to store the resampled stack
                    zarr_rot, zarr_mip = get_arrays_to_store(main_zarr_path, attrs['view'], ch)

                    if v == 0:
                        zarr_rot[0][counter, :, :, :] = data_rotate
                        zarr_mip[0][counter, :, :] = data_mip
                    else:
                        zarr_rot[1][counter, :, :, :] = data_rotate
                        zarr_mip[1][counter, :, :] = data_mip

                counter = counter + 1
                print("time elapsed for current timepoint:", time.time() - start_tp)


def _get_attributes(zarr_path):
    """get the attributes stored in the zarr folder"""
    root = zarr.open(zarr_path, mode='r')
    attrs = {}
    for name in root.attrs:
        attrs[name] = root.attrs[name]
    return attrs


def get_arrays_to_store(path, num_view, ch):
    """get the zarr arrays to store resampled data and their MIPs"""
    data_rot = []
    data_mip = []
    root_all_tp = zarr.open(path, mode='rw')
    for view in range(num_view):
        name = f'v{view}c{ch}_rot'
        d_rot = root_all_tp[name][name]
        name = f'v{view}c{ch}_mip'
        d_mip = root_all_tp[name][name]
        data_rot.append(d_rot)
        data_mip.append(d_mip)
    return data_rot, data_mip


def readout_zarr_data(root, num_view, ch, ends=''):
    """readout the zarr data. Given the parameters, return the zarr arrays"""
    data = []
    for view in range(num_view):
        name = f'v{view}c{ch}' + ends
        data.append(root[name][name])
    return data


def create_arrays_in_zarr(root_all_tp, attrs, nb_tp_sub, nz, ny, nx, compressor):
    """create the data arrays in the zarr folder to hold the resampled data and its mip"""
    nb_view, nb_channel = attrs['view'], attrs['channel']
    for ch in range(nb_channel):
        for v in range(nb_view):
            name = f'v{v}c{ch}_rot'
            group = root_all_tp.create_group(name)
            group.zeros(name,
                        shape=(nb_tp_sub, nz, ny, nx),
                        chunks=(1, 128, 1024, 1024),
                        dtype='uint16',
                        compressor=compressor)
            name = f'v{v}c{ch}_mip'
            group = root_all_tp.create_group(name)
            group.zeros(name,
                        shape=(nb_tp_sub, ny, nx),
                        chunks=(64, None, None),
                        dtype='uint16',
                        compressor=compressor)
    print('zarr arrays after creation', root_all_tp.tree())

    save_attributes(root_all_tp, attrs)


def save_attributes(root, attrs):
    """save the atrributes in the zarr folder"""
    for name in attrs:
        root.attrs[name] = attrs[name]


def find_all_files_ends_with(path, ends='.zarr'):
    """find all the files with ends in the directory (only current directory) given its path, natually sort the files and
    return the paths of all the file in it

    :param path: String, directory path
    :return: list_of_files: dictionary, {<filename>: <fullpath>}
    """
    files = [file for file in os.listdir(path) if file.endswith(ends)]

    sorted_files = natsorted(files)
    list_of_files = [os.sep.join([path, file]) for file in sorted_files]
    return list_of_files


def process_v0(data, t, scope_settings, num_split):
    """process one timepoint for data from view0"""
    data_1tp = data[t, :, :, :]

    data_gpu = cp.asarray(data_1tp)
    data_gpu = cp.flip(data_gpu, axis=0)

    # rotate the data
    data_rot = cp.rot90(data_gpu, k=1, axes=(1, 2))
    data_rot = cp.array(data_rot, copy=True, order='C')
    del data_gpu

    # deskew and rotate
    data_rotate, data_mip = resampling_vertical_cupy_split(data_rot,
                                                           dx=scope_settings['res'],
                                                           dz=scope_settings['dz'],
                                                           angle=scope_settings['angle'],
                                                           num_split=num_split)
    del data_rot

    # flip along axis x
    data_rotate = np.flip(data_rotate, axis=2)
    data_mip = np.flip(data_mip, axis=1)
    return data_rotate, data_mip


def process_v1(data, t, scope_settings, num_split):
    """process one timepoint for data from view1"""
    data_1tp = data[t, :, :, :]
    data_gpu = cp.asarray(data_1tp)

    # rotate the data
    data_rot = cp.rot90(data_gpu, k=1, axes=(1, 2))
    data_rot = cp.array(data_rot, copy=True, order='C')
    del data_gpu

    # deskew and rotate
    start = time.time()
    data_rotate, data_mip = resampling_vertical_cupy_split(data_rot,
                                                           dx=scope_settings['res'],
                                                           dz=scope_settings['dz'],
                                                           angle=scope_settings['angle'],
                                                           num_split=num_split)
    del data_rot

    return data_rotate, data_mip


def resampling_vertical_cupy_split(data, dz=1.0, dx=0.2, angle=45, num_split=4):
    data_gpu_splits = cp.array_split(data, num_split, axis=1)
    for k in range(num_split):
        data_resampled = resampling_vertical_cupy(data_gpu_splits[k], dz, dx, angle=45)
        if k == 0:
            output = cp.asnumpy(data_resampled)
            data_mip = cp.max(data_resampled, axis=0)  # get mip
            output_mip = cp.asnumpy(data_mip)
        else:
            output = np.concatenate((output, cp.asnumpy(data_resampled)), axis=1)
            data_mip = cp.max(data_resampled, axis=0)  # get mip
            output_mip = np.concatenate((output_mip, cp.asnumpy(data_mip)), axis=0)
    del data_resampled, data_mip, data
    return output, output_mip


def resampling_vertical_cupy(data, dz=1.0, dx=0.2, angle=45):
    """resampling of the data by interpolation along vertical direction.
    Here we assume the dz is integer multiple of dx * cos(angle * pi / 180),
    one can also pre interpolate the data within along the z' axis if this is not the case
     :param
     data   : ndarray in cupy, 3D stack
     dz     : float, scanning step (stage or galvo scanning step, not the same as the distance between the slices)
     dx     : float, pixel size of the camera
     angle  : float, incident angle of the light shee, angle between the light sheet and the optical axis
     num_split  : number of splits to break down the data into pieces (along y, axis=2) to fit into the memory of GPU"""
    (nz, ny, nx) = data.shape

    zres = dz * np.sin(angle * np.pi / 180)
    xres = dx * np.cos(angle * np.pi / 180)

    resample_factor = dz / xres
    resample_factor_int = int(round(resample_factor))

    nz_new, ny_new, nx_new = len(range(0, nx, resample_factor_int)), ny, nx + nz * resample_factor_int
    data_reassign = cp.zeros((nz_new, ny_new, nx_new), cp.int16)

    for x in range(nx):
        x_start = x
        x_end = nz * resample_factor_int + x
        data_reassign[x // resample_factor_int, :, x_start:x_end:resample_factor_int] = data[:, :, x].T
    del data

    # rescale the data, interpolate along z
    data_rescale = cp_ndimage.zoom(data_reassign, (resample_factor_int, 1, 1))
    del data_reassign

    data_interp = cp.zeros((nz_new, ny_new, nx_new), cp.int16)

    for z in range(nz_new):
        for k in range(resample_factor_int):
            data_interp[z, :, k::resample_factor_int] = \
                data_rescale[z * resample_factor_int - k, :, k::resample_factor_int]
    del data_rescale

    # rescale the data, to have voxel the same along x an y;
    # remove the first z slice which has artifacts due to resampling
    data_final = cp_ndimage.zoom(data_interp[1:], (1, 1, xres / dx))
    del data_interp
    return data_final
