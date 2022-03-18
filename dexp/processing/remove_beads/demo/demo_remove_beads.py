import click
import numpy as np
from arbol import asection

from dexp.processing.remove_beads.beadsremover import remove_beads_by_threshold
from dexp.utils.backends import CupyBackend


@click.command()
@click.argument("input_path", nargs=1)
@click.option("--zero-level", "-zl", default=40, help="Camera noise zero level.")
def main(input_path, zero_level):
    import napari

    viewer = napari.Viewer()
    (layer,) = viewer.open(input_path, name="original data")

    with CupyBackend() as bk:
        image = bk.to_backend(layer.data)
        image = np.clip(image, zero_level, None)
        image -= zero_level

        with asection("Removing beads"):
            clean = remove_beads_by_threshold(image)
            clean = bk.to_numpy(clean)

        viewer.add_image(clean, name="without beads")

    napari.run()


if __name__ == "__main__":
    main()
