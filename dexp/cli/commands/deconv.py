from time import time

import click

from dexp.cli.main import _get_dataset_from_path, _get_folder_name_without_end_slash, _parse_slicing, _default_store, _default_codec, _default_clevel
from dexp.datasets.operations.deconv import dataset_deconv


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None , help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Store: ‘dir’, ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)
@click.option('--workers', '-k', default=1, help='Number of worker threads to spawn, recommended: 1 (unless you know what you are doing)', show_default=True)
@click.option('--method', '-m', type=str, default='lr', help='Deconvolution method: for now only lr (Lucy Richardson)', show_default=True)
@click.option('--iterations', '-i', type=int, default=15, help='Number of deconvolution iterations. More iterations takes longer, will be sharper, but might also be potentially more noisy depending on method.', show_default=True)
@click.option('--maxcorrection', '-mc', type=int, default=2, help='Max correction in folds per iteration. Noisy datasets benefit from mc=2 (recommended), for noiseless datasets you can push to mc=8 or even more.', show_default=True)
@click.option('--power', '-pw', type=float, default=1.5, help='Correction exponent, default for standard LR is 1, set to 1.5 for acceleration.', show_default=True)
@click.option('--dxy', '-dxy', type=float, default=0.485, help='Voxel size along x and y in microns', show_default=True)
@click.option('--dz', '-dz', type=float, default=4*0.485, help='Voxel size along z in microns', show_default=True)
@click.option('--xysize', '-sxy', type=int, default=17, help='Voxel size along xy in microns', show_default=True)
@click.option('--zsize', '-sz', type=int, default=31, help='Voxel size along z in microns', show_default=True)
@click.option('--downscalexy2', '-d', is_flag=False, help='Downscales along x and y for faster deconvolution (but worse quality of course)', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deconv(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, workers, method, iterations, maxcorrection, power, dxy, dz, xysize, zsize, downscalexy2, check):
    input_dataset = _get_dataset_from_path(input_path)

    print(f"Available Channels: {input_dataset.channels()}")
    for channel in input_dataset.channels():
        print(f"Channel '{channel}' shape: {input_dataset.shape(channel)}")

    if output_path is None or not output_path.strip():
        output_path = _get_folder_name_without_end_slash(input_path) + '.zarr'

    slicing = _parse_slicing(slicing)
    print(f"Requested slicing: {slicing} ")

    print(f"Requested channel(s)  {channels if channels else '--All--'} ")
    if not channels is None:
        channels = channels.split(',')
    print(f"Selected channel(s): '{channels}' and slice: {slicing}")

    print("Fusing dataset.")
    print(f"Saving dataset to: {output_path} with zarr format... ")
    time_start = time()
    dataset_deconv(input_dataset,
           output_path,
           channels=channels,
           slicing=slicing,
           store=store,
           compression=codec,
           compression_level=clevel,
           overwrite=overwrite,
           workers=workers,
           method=method,
           num_iterations=iterations,
           max_correction=maxcorrection,
           power=power,
           dxy=dxy,
           dz=dz,
           xy_size=xysize,
           z_size=zsize,
           downscalexy2=downscalexy2,
           check=check
           )


    time_stop = time()
    print(f"Elapsed time to write dataset: {time_stop - time_start} seconds")
    input_dataset.close()
    print("Done!")
