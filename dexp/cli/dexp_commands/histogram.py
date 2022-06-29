import click

from dexp.cli.parsing import channels_option, device_option, input_dataset_argument
from dexp.datasets.operations.histogram import dataset_histogram


@click.command()
@input_dataset_argument()
@channels_option()
@device_option()
@click.option("--output-directory", "-o", default="histogram", show_default=True, type=str)
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
def histogram(input_dataset, output_directory, channels, device, minimum_count, maximum_count):
    """Computes histogram of selected channel."""
    dataset_histogram(
        dataset=input_dataset,
        output_dir=output_directory,
        channels=channels,
        device=device,
        minimum_count=minimum_count,
        maximum_count=maximum_count,
    )
