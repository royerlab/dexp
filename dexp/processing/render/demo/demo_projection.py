from arbol import asection

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.render.projection import rgb_project
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data


def demo_projection_numpy():
    with NumpyBackend():
        demo_projection()


def demo_projection_cupy():
    try:
        with CupyBackend():
            demo_projection(length_xy=512)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_projection(length_xy=120):
    with asection("generate data"):
        _, _, image = generate_nuclei_background_data(add_noise=False,
                                                      length_xy=length_xy,
                                                      length_z_factor=1,
                                                      background_stength=0.001,
                                                      sphere=True,
                                                      zoom=2)

    with asection("max_projection"):
        max_projection = rgb_project(image, mode='max', attenuation=0.1)

    with asection("color_max_projection"):
        color_max_projection = rgb_project(image, mode='colormax', attenuation=0.1)

    with asection("color_max_projection_bottom"):
        color_max_projection_bottom = rgb_project(image, mode='colormax', attenuation=0.1, dir=+1)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(max_projection), name='max_projection', rgb=True)
        viewer.add_image(_c(color_max_projection), name='color_max_projection', rgb=True)
        viewer.add_image(_c(color_max_projection_bottom), name='color_max_projection_bottom', rgb=True)


if __name__ == "__main__":
    # demo_fusion_cupy()
    demo_projection_numpy()
