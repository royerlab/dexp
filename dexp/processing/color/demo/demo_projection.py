import numpy
from arbol import asection

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.color.projection import project_image
from dexp.processing.color.projection_legend import depth_color_scale_legend
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_projection_numpy():
    # TODO: projection behaves differently in numpy ?!? back and front views look the same ?!!
    with NumpyBackend():
        demo_projection(length_xy=150)


def demo_projection_cupy():
    try:
        with CupyBackend():
            demo_projection(length_xy=320)
        return True
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")
        return False


def demo_projection(length_xy=120, display=True):
    with asection("generate data"):
        _, _, image = generate_nuclei_background_data(
            add_noise=False,
            length_xy=length_xy,
            length_z_factor=1,
            background_strength=0.001,
            sphere=True,
            radius=0.7,
            zoom=2,
            dtype=numpy.uint16,
        )

        # we place an indicator to know where is the depth 0, will appear at the bottom right of the image:
        image[0, 2 * length_xy - 20 : 2 * length_xy, 2 * length_xy - 20 : 2 * length_xy] = 300

        # we place an indicator to know where is the depth max, will appear at the top right of the image:
        image[image.shape[0] - 1, 0:20, 2 * length_xy - 20 : 2 * length_xy] = 300

        # we place an indicator on top of the sample:
        image[image.shape[0] - 5 : image.shape[0] - 1, :, length_xy - 10 : length_xy + 10] = 300

    with asection("max_projection"):
        max_projection = project_image(image, mode="max")

    with asection("max_projection_att"):
        max_projection_att = project_image(image, mode="max", attenuation=0.05)

    with asection("max_color_projection"):
        max_color_projection = project_image(
            image, mode="maxcolor", attenuation=0.05, cmap="rainbow", dlim=(0.3, 0.7), legend_size=0.2
        )

    with asection("max_color_projection_bottom"):
        max_color_projection_bottom = project_image(
            image, mode="maxcolor", attenuation=0.05, cmap="rainbow", dlim=(0.3, 0.7), dir=+1, legend_size=0.2
        )

    with asection("color_max_projection"):
        color_max_projection = project_image(
            image, mode="colormax", attenuation=0.1, cmap="rainbow", dlim=(0.3, 0.7), legend_size=0.2
        )

    with asection("color_max_projection_bottom"):
        color_max_projection_bottom = project_image(
            image, mode="colormax", attenuation=0.1, cmap="rainbow", dlim=(0.3, 0.7), dir=+1, legend_size=0.2
        )

    with asection("color_max_projection_transparent"):
        color_max_projection_transparent = project_image(
            image,
            mode="colormax",
            attenuation=0.1,
            cmap="rainbow",
            dlim=(0.3, 0.7),
            legend_size=0.2,
            transparency=True,
        )

    color_legend = depth_color_scale_legend(cmap="rainbow", start=0, end=100, title="color-coded depth (μm)", size=1)

    # image_shifted = sp.ndimage.shift(image, shift=(50, 70, -23), order=1, mode='nearest')

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name="image")
            viewer.add_image(_c(color_legend), name="color_legend", rgb=True)
            viewer.add_image(_c(max_projection), name="max_projection", rgb=True)
            viewer.add_image(_c(max_projection_att), name="max_projection_att", rgb=True)
            viewer.add_image(_c(color_max_projection), name="color_max_projection", rgb=True)
            viewer.add_image(_c(color_max_projection_bottom), name="color_max_projection_bottom", rgb=True)
            viewer.add_image(_c(max_color_projection), name="max_color_projection", rgb=True)
            viewer.add_image(_c(max_color_projection_bottom), name="max_color_projection_bottom", rgb=True)
            viewer.add_image(_c(color_max_projection_transparent), name="color_max_projection_transparent", rgb=True)
            viewer.grid.enabled = True


if __name__ == "__main__":
    if not demo_projection_cupy():
        demo_projection_numpy()
