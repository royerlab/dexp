import click
from arbol import asection

from dexp.cli.parsing import (
    channels_callback,
    input_dataset_argument,
    multi_devices_option,
    output_dataset_options,
    slicing_option,
)
from dexp.datasets.operations.segment import dataset_segment


@click.command()
@input_dataset_argument()
@output_dataset_options()
@multi_devices_option()
@slicing_option()
@click.option(
    "--detection-channels",
    "-dc",
    default=None,
    help="list of channels used to detect cells.",
    callback=channels_callback,
)
@click.option(
    "--features-channels",
    "-fc",
    default=None,
    help="list of channels for cell features extraction.",
    callback=channels_callback,
)
@click.option("--out-channel", "-oc", default="Segments", help="Output channel name.", show_default=True)
@click.option("--z-scale", "-z", type=int, default=1, help="Anisotropy over z axis.")
@click.option(
    "--area-threshold",
    "-a",
    type=click.FloatRange(min=0),
    default=1e4,
    help="Parameter for cell detection, a smaller values will detect fewer segments.",
    show_default=True,
)
@click.option("--minimum-area", "-m", type=float, default=500, help="Minimum area (sum of pixels) per segment.")
@click.option(
    "--h-minima",
    "-h",
    type=float,
    default=1,
    help="Parameter to adjust the number of segments, smaller results in more segments.",
    show_default=True,
)
@click.option(
    "--compactness",
    "-cmp",
    type=click.FloatRange(min=0.0),
    default=0,
    help="Cell compactness (convexity) penalization, it penalizes non-convex shapes.",
    show_default=True,
)
@click.option(
    "--gamma",
    "-g",
    type=click.FloatRange(min=0.0),
    default=1.0,
    help="Gamma parameter to equalize (exponent) the image intensities.",
    show_default=True,
)
@click.option(
    "--use-edt",
    "-e/-ne",
    is_flag=True,
    default=True,
    help="Use Euclidean Distance Transform (EDT) or image intensity for segmentation.",
    show_default=True,
)
def segment(**kwargs):
    """Segment cells of a combination of multiple detection channels and extract features of each segment from the feature channels."""
    with asection(
        f"Segmenting {kwargs['input_dataset'].path} with {kwargs['detection_channels']} and extracting features"
        f"of {kwargs['features_channels']}, results saved to to {kwargs['output_dataset'].path}"
    ):
        dataset_segment(
            **kwargs,
        )

    kwargs["input_dataset"].close()
    kwargs["output_dataset"].close()
