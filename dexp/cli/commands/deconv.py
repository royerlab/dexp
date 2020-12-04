import click

from dexp.cli.main import _default_store, _default_codec, _default_clevel
from dexp.cli.utils import _parse_channels, _get_dataset_from_path, _get_output_path, _parse_slicing, _parse_devices
from dexp.datasets.operations.deconv import dataset_deconv
from dexp.utils.timeit import timeit


@click.command()
@click.argument('input_path')  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='list of channels, all channels when ommited.')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='to force overwrite of target', show_default=True)
@click.option('--chunksize', '-cs', type=int, default=512, help='Chunk size for tiled computation', show_default=True)
@click.option('--method', '-m', type=str, default='lr', help='Deconvolution method: for now only lr (Lucy Richardson)', show_default=True)
@click.option('--iterations', '-i', type=int, default=20, help='Number of deconvolution iterations. More iterations takes longer, will be sharper, but might also be potentially more noisy depending on method.', show_default=True)
@click.option('--maxcorrection', '-mc', type=int, default=16, help='Max correction in folds per iteration.',
              show_default=True)
@click.option('--power', '-pw', type=float, default=1.0, help='Correction exponent, default for standard LR is 1, set to >1 for acceleration.', show_default=True)
@click.option('--blindspot', '-bs', type=int, default=0, help='Blindspot based noise reduction. Provide size of kernel to use, must be an odd number: 3(recommended), 5, 7. 0 means no blindspot. ', show_default=True)
@click.option('--backprojection', '-bp', type=str, default='tpsf', help='Back projection operator, can be: ‘tpsf’ (transposed PSF = classic) or ‘wb’ (Wiener-Butterworth =  accelerated) ', show_default=True)
@click.option('--objective', '-obj', type=str, default='nikon16x08na', help='Microscope objective to use for computing psf, can be: nikon16x08na or olympus20x10na', show_default=True)
@click.option('--dxy', '-dxy', type=float, default=0.485, help='Voxel size along x and y in microns', show_default=True)
@click.option('--dz', '-dz', type=float, default=4 * 0.485, help='Voxel size along z in microns', show_default=True)
@click.option('--xysize', '-sxy', type=int, default=17, help='PSF size along xy in voxels', show_default=True)
@click.option('--zsize', '-sz', type=int, default=17, help='PSF size along z in voxels', show_default=True)
@click.option('--downscalexy2', '-d', is_flag=False, help='Downscales along x and y for faster deconvolution (but worse quality of course)', show_default=True)  #
@click.option('--workers', '-k', type=int, default=-1, help='Number of worker threads to spawn, if -1 then num workers = num devices', show_default=True)
@click.option('--devices', '-d', type=str, default='0', help='Sets the CUDA devices id, e.g. 0,1,2', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def deconv(input_path, output_path, channels, slicing, store, codec, clevel, overwrite, chunksize,
           method, iterations, maxcorrection, power, blindspot, backprojection, objective, dxy, dz, xysize, zsize, downscalexy2,
           workers, devices, check):
    input_dataset = _get_dataset_from_path(input_path)
    output_path = _get_output_path(input_path, output_path, ".deconv")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)
    devices = _parse_devices(devices)

    with timeit("deconvolution"):
        print("Fusing dataset.")
        dataset_deconv(input_dataset,
                       output_path,
                       channels=channels,
                       slicing=slicing,
                       store=store,
                       compression=codec,
                       compression_level=clevel,
                       overwrite=overwrite,
                       chunksize=chunksize,
                       method=method,
                       num_iterations=iterations,
                       max_correction=maxcorrection,
                       power=power,
                       blind_spot=blindspot,
                       back_projection=backprojection,
                       objective=objective,
                       dxy=dxy,
                       dz=dz,
                       xy_size=xysize,
                       z_size=zsize,
                       downscalexy2=downscalexy2,
                       workers=workers,
                       devices=devices,
                       check=check
                       )

    input_dataset.close()
    print("Done!")
