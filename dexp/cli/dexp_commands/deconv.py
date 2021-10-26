import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_store, _default_codec, _default_clevel, _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.deconv import dataset_deconv


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)
@click.option('--tilesize', '-ts', type=int, default=512, help='Tile size for tiled computation', show_default=True)
@click.option('--method', '-m', type=str, default='lr', help='Deconvolution method: for now only lr (Lucy Richardson)', show_default=True)
@click.option('--iterations', '-i', type=int, default=None, help='Number of deconvolution iterations. More iterations takes longer, will be sharper, but might also be potentially more noisy depending on method. '
                                                                 'The default number of iterations depends on the other parameters, in particular it depends on the choice of backprojection operator. For ‘wb’ as little as 3 iterations suffice. ',
              show_default=True)
@click.option('--maxcorrection', '-mc', type=int, default=None, help='Max correction in folds per iteration. By default there is no limit',
              show_default=True)
@click.option('--power', '-pw', type=float, default=1.0, help='Correction exponent, default for standard LR is 1, set to >1 for acceleration.', show_default=True)
@click.option('--blindspot', '-bs', type=int, default=0, help='Blindspot based noise reduction. Provide size of kernel to use, must be an odd number: 3(recommended), 5, 7. 0 means no blindspot. ', show_default=True)
@click.option('--backprojection', '-bp', type=str, default='tpsf', help='Back projection operator, can be: ‘tpsf’ (transposed PSF = classic) or ‘wb’ (Wiener-Butterworth =  accelerated) ', show_default=True)
@click.option('--wb-order', '-wbo', type=int, default=5, help='Wiener-Butterworth order parameter, a higher `n` makes the filter transition slope closer to a hard cutoff, causing the ringing artifacts in the spatial domain.'
                                                              'In contrast, a lower order `n` makes the transition slope gentler and sacrifice some spectral amplitude at spatial frequencies close to the cutoff.', show_default=True)
@click.option('--objective', '-obj', type=str, default='nikon16x08na', help='Microscope objective to use for computing psf, can be: nikon16x08na, olympus20x10na, or path to file', show_default=True)
@click.option('--numape', '-na', type=float, default=None, help='Overrides NA value for objective.', show_default=True)
@click.option('--dxy', '-dxy', type=float, default=0.485, help='Voxel size along x and y in microns', show_default=True)
@click.option('--dz', '-dz', type=float, default=4 * 0.485, help='Voxel size along z in microns', show_default=True)
@click.option('--xysize', '-sxy', type=int, default=31, help='PSF size along xy in voxels', show_default=True)
@click.option('--zsize', '-sz', type=int, default=31, help='PSF size along z in voxels', show_default=True)
@click.option('--showpsf', '-sp', is_flag=True, help='Show point spread function (PSF) with napari', show_default=True)
@click.option('--scaling', '-sc', type=str, default='1,1,1', help='Scales input image along the three axis: sz,sy,sx (numpy order). For example: 2,1,1 upscales along z by a factor 2', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2 or ‘all’', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deconv(input_paths,
           output_path,
           channels,
           slicing,
           store,
           codec,
           clevel,
           overwrite,
           tilesize,
           method,
           iterations,
           maxcorrection,
           power,
           blindspot,
           backprojection,
           wb_order,
           objective,
           numape,
           dxy, dz,
           xysize, zsize,
           showpsf,
           scaling,
           workers,
           workersbackend,
           devices,
           check):
    """ Deconvolves all or selected channels of a dataset.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_deconv")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    if ',' in scaling:
        scaling = tuple(float(v) for v in scaling.split(','))

    with asection(f"Deconvolving dataset: {input_paths}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        dataset_deconv(input_dataset,
                       output_path,
                       channels=channels,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       tilesize=tilesize,
                       method=method,
                       num_iterations=iterations,
                       max_correction=maxcorrection,
                       power=power,
                       blind_spot=blindspot,
                       back_projection=backprojection,
                       wb_order=wb_order,
                       psf_objective=objective,
                       psf_na=numape,
                       psf_dxy=dxy,
                       psf_dz=dz,
                       psf_xy_size=xysize,
                       psf_z_size=zsize,
                       psf_show=showpsf,
                       scaling=scaling,
                       workers=workers,
                       workersbackend=workersbackend,
                       devices=devices,
                       check=check
                       )

        input_dataset.close()
        aprint("Done!")
