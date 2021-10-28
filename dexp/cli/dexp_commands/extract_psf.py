import click
from arbol.arbol import aprint, asection

from dexp.cli.parsing import _parse_channels, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.extract_psf import dataset_extract_psf


@click.command(name='extract-psf')
@click.argument('input_paths', nargs=-1, required=True)
@click.option('--out-prefix-path', '-o', default='psf_', help='Output PSF file prefix')
@click.option('--channels', '-c', default=None,
             help='list of channels to extract the PSF, each channel will have their separater PSF (i.e., psf_c0.npy, psf_c1.npy')
@click.option('--slicing', '-s', default=None, help='dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ')
@click.option('--peak-threshold', '-pt', type=int, default=500, show_default=True,
              help='Peak valeu threshold for object (PSF) detection. Lower values are less consertaive and will detect more objects.')
@click.option('--similarity-threshold', '-st', type=float, default=0.5, show_default=True,
              help='Threshold of PSF selection given the similarity (cosine distance) to median PSF.')
@click.option('--psf_size', '-ps', type=int, default=35, show_default=True, help='Size (shape) of the PSF')
@click.option('--device', '-d', type=int, default=0, help='Sets the CUDA device id, e.g. 0,1,2. It works for only a single device!', show_default=True)
@click.option('--verbose', '-v', type=bool, is_flag=True, default=False, help='Flag to display intermediated results.')
def extract_psf(input_paths,
                out_prefix_path,
                channels,
                slicing,
                peak_threshold,
                similarity_threshold,
                psf_size,
                device,
                verbose,
                ):
    """Extracts the PSF from beads.
    """
    input_dataset, input_paths = glob_datasets(input_paths)
    slicing = _parse_slicing(slicing)
    channels = _parse_channels(input_dataset, channels)

    with asection(f"Extracting PSF of dataset: {input_paths}, saving it with prefix: {out_prefix_path}, for channels: {channels}, slicing: {slicing} "):
        aprint(f"Device used: {device}")
        dataset_extract_psf(dataset=input_dataset,
                            dest_path=out_prefix_path,
                            channels=channels,
                            slicing=slicing,
                            peak_threshold=peak_threshold,
                            similarity_threshold=similarity_threshold,
                            psf_size=psf_size,
                            verbose=verbose,
                            device=device)
