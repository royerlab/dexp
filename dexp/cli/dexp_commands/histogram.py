import click

from dexp.cli.parsing import _parse_channels, _parse_slicing
from dexp.datasets.open_dataset import glob_datasets
from dexp.datasets.operations.histogram import dataset_histogram


@click.command()
@click.argument("input-paths", nargs=-1)
@click.option("--output-directory", "-o", default="histogram", show_default=True, type=str)
@click.option("--channels", "-c", default=None, help="list of channels, all channels when ommited.")
@click.option(
    "--slicing",
    "-s",
    default=None,
    help="dataset slice (TZYX), e.g. [0:5] (first five stacks) [:,0:100] (cropping in z) ",
)
@click.option("--device", "-d", type=int, default=0, show_default=True)
def histogram(input_paths, output_directory, channels, slicing, device):

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)

    if slicing is not None:
        input_dataset.set_slicing(slicing)

    dataset_histogram(input_dataset, output_directory, channels, device)
