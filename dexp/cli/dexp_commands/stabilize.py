import click
from arbol.arbol import aprint, asection

from dexp.cli.defaults import _default_store, _default_codec, _default_clevel, _default_workers_backend
from dexp.cli.parsing import _parse_channels, _get_output_path, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.stabilize import dataset_stabilize


@click.command()
@click.argument('input_paths', nargs=-1)  # ,  help='input path'
@click.option('--output_path', '-o')  # , help='output path'
@click.option('--channels', '-c', default=None, help='List of channels, all channels when omitted.')
@click.option('--reference-channel', '-rc', default=None, help='Reference channel for single stabilization model computation.')
@click.option('--slicing', '-s', default=None, help='Dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--store', '-st', default=_default_store, help='Zarr store: ‘dir’, ‘ndir’, or ‘zip’', show_default=True)
@click.option('--codec', '-z', default=_default_codec, help='Compression codec: ‘zstd’, ‘blosclz’, ‘lz4’, ‘lz4hc’, ‘zlib’ or ‘snappy’ ', show_default=True)
@click.option('--clevel', '-l', type=int, default=_default_clevel, help='Compression level', show_default=True)
@click.option('--overwrite', '-w', is_flag=True, help='Forces overwrite of target', show_default=True)
@click.option('--maxrange', '-mr', type=int, default=7, help='Maximal distance, in time points, between pairs of images to registrate.', show_default=True)
@click.option('--minconfidence', '-mc', type=float, default=0.5, help='Minimal confidence for registration parameters, if below that level the registration parameters for previous time points is used.', show_default=True)  #
@click.option('--com/--no-com', type=bool, default=False, help='Enable center of mass fallback when standard registration fails.', show_default=True)
@click.option('--quantile', '-q', type=float, default=0.5, help='Quantile to cut-off background in center-of-mass calculation.', show_default=True)
@click.option('--tolerance', '-t', type=float, default=1e-7, help='Tolerance for linear solver.', show_default=True)
@click.option('--ordererror', '-oe', type=float, default=2.0, help='Order for linear solver error term.', show_default=True)
@click.option('--orderreg', '-or', type=float, default=1.0, help='Order for linear solver regularisation term.', show_default=True)
@click.option('--alphareg', '-or', type=float, default=1e-4, help='Multiplicative coefficient for regularisation term.', show_default=True)
@click.option('--pcsigma', '-rs', type=float, default=2, help='Sigma for Gaussian smoothing of phase correlogram, zero to disable.', show_default=True)
@click.option('--dsigma', '-ds', type=float, default=1.5, help='Sigma for Gaussian smoothing (crude denoising) of input images, zero to disable.', show_default=True)
@click.option('--logcomp', '-lc', type=bool, default=True, help='Applies the function log1p to the images to compress high-intensities (usefull when very (too) bright structures are present in the images, such as beads.', show_default=True)
@click.option('--edgefilter', '-ef', type=bool, default=False, help='Applies sobel edge filter to input images.', show_default=True)
@click.option('--detrend', '-dt', type=bool, is_flag=True, default=False, help='Remove linear trend from stabilization result', show_default=True)
@click.option('--maxproj/--no-maxproj', '-mp/-nmp', type=bool, default=True, help='Registers using only the maximum intensity projection from each stack.', show_default=True)
@click.option('--workers', '-k', type=int, default=-4, help='Number of worker threads to spawn. Negative numbers n correspond to: number_of _cores / |n|. Be careful, starting two many workers is know to cause trouble (unfortunately unclear why!).',
              show_default=True)
@click.option('--workersbackend', '-wkb', type=str, default=_default_workers_backend, help='What backend to spawn workers with, can be ‘loky’ (multi-process) or ‘threading’ (multi-thread) ', show_default=True)  #
@click.option('--device', '-d', type=int, default=0, help='Sets the CUDA devices id, e.g. 0,1,2', show_default=True)  #
@click.option('--check', '-ck', default=True, help='Checking integrity of written file.', show_default=True)  #
def stabilize(input_paths,
              output_path,
              channels,
              reference_channel,
              slicing,
              store,
              codec,
              clevel,
              overwrite,
              maxrange,
              minconfidence,
              com,
              quantile,
              tolerance,
              ordererror,
              orderreg,
              alphareg,
              pcsigma,
              dsigma,
              logcomp,
              edgefilter,
              detrend,
              maxproj,
              workers,
              workersbackend,
              device,
              check):
    """ Stabilises dataset against translations across time.
    """

    input_dataset, input_paths = glob_datasets(input_paths)
    output_path = _get_output_path(input_paths[0], output_path, "_stabilized")

    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"Stabilizing dataset(s): {input_paths}, saving it at: {output_path}, for channels: {channels}, slicing: {slicing} "):
        dataset_stabilize(input_dataset,
                          output_path,
                          channels=channels,
                          reference_channel=reference_channel,
                          slicing=slicing,
                          zarr_store=store,
                          compression_codec=codec,
                          compression_level=clevel,
                          overwrite=overwrite,
                          max_range=maxrange,
                          min_confidence=minconfidence,
                          enable_com=com,
                          quantile=quantile,
                          tolerance=tolerance,
                          order_error=ordererror,
                          order_reg=orderreg,
                          alpha_reg=alphareg,
                          phase_correlogram_sigma=pcsigma,
                          denoise_input_sigma=dsigma,
                          log_compression=logcomp,
                          edge_filter=edgefilter,
                          detrend=detrend,
                          maxproj=maxproj,
                          workers=workers,
                          workers_backend=workersbackend,
                          device=device,
                          check=check,
                          debug_output='stabilization'
                          )

        input_dataset.close()
        aprint("Done!")
