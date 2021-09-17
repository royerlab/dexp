import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_codec, _default_store, _default_clevel, _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.deskew import dataset_deskew


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels for the view in standard order for the microscope type (C0L0, C0L1, C1L0, C1L1,...)')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')  #
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ')
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)  # , help='dataset slice'
@click.option('--mode', '-m', type=str, default='yang', help="Deskew algorithm: 'yang' or 'classic'. ", show_default=True)  #
@click.option('--deltax', '-xx', type=float, default=None, help='Pixel size of the camera', show_default=True)  #
@click.option('--deltaz', '-zz', type=float, default=None, help='Scanning step (stage or galvo scanning step, not the same as the distance between the slices)',
              show_default=True)  #
@click.option('--angle', '-a', type=float, default=None, help='Incident angle of the light sheet, angle between the light sheet and the optical axis in degrees', show_default=True)  #
@click.option('--flips', '-fl', type=str, default=None, help='Flips image to deskew in the opposite orientation (True for view 0 and False for view 1)', show_default=True)  #
@click.option('--camorientation', '-co', type=int, default=0, help='Camera orientation correction expressed as a number of 90 deg rotations to be performed per 2D image in stack -- if required.', show_default=True)  #
@click.option('--depthaxis', '-za', type=int, default=0, help='Depth axis.', show_default=True)  #
@click.option('--lateralaxis', '-xa', type=int, default=1, help='Lateral axis.', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)  #
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deskew(input_paths,
           output_path,
           channels,
           slicing,
           store,
           codec,
           clevel,
           overwrite,
           mode,
           deltax,
           deltaz,
           angle,
           flips,
           camorientation,
           depthaxis,
           lateralaxis,
           workers,
           workersbackend,
           devices,
           check):
    """ Deskews all or selected channels of a dataset.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_deskew")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    if flips is None:
        flips = (False, )*len(channels)
    else:
        flips = tuple(bool(s) for s in flips.split(','))

    with asection(f"Deskewing dataset: {input_paths}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        aprint(f"Devices used: {devices}, workers: {workers} ")
        dataset_deskew(input_dataset,
                       output_path,
                       channels=channels,
                       slicing=slicing,
                       dx=deltax,
                       dz=deltaz,
                       angle=angle,
                       flips=flips,
                       camera_orientation=camorientation,
                       depth_axis=depthaxis,
                       lateral_axis=lateralaxis,
                       mode=mode,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       workers=workers,
                       workersbackend=workersbackend,
                       devices=devices,
                       check=check)

        input_dataset.close()
        aprint("Done!")
