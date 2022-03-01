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
@click.option(
    "--minimum-count",
    "-m",
    default=5,
    show_default=True,
    type=int,
    help="Minimum count on the histogram used to select upper limit, values lower than this are not considered.",
)
@click.option(
    "--maximum-count",
    "-M",
    default=None,
    show_default=True,
    type=float,
    help="Maximum count to clip histogram and improve visualization.",
)
def histogram(input_paths, output_directory, channels, slicing, device, minimum_count, maximum_count):

    input_dataset, input_paths = glob_datasets(input_paths)
    channels = _parse_channels(input_dataset, channels)
    slicing = _parse_slicing(slicing)

    if slicing is not None:
        input_dataset.set_slicing(slicing)

    dataset_histogram(
        dataset=input_dataset,
        output_dir=output_directory,
        channels=channels,
        device=device,
        minimum_count=minimum_count,
        maximum_count=maximum_count,
    )
