import numpy
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
            demo_projection(length_xy=320)
    except (ModuleNotFoundError, NotImplementedError):
        print("Cupy module not found! ignored!")


def demo_projection(length_xy=120):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    with asection("generate data"):
        _, _, image = generate_nuclei_background_data(add_noise=False,
                                                      length_xy=length_xy,
                                                      length_z_factor=1,
                                                      background_stength=0.001,
                                                      sphere=True,
                                                      radius=0.7,
                                                      zoom=2,
                                                      dtype=numpy.uint16)

    with asection("max_projection"):
        max_projection = rgb_project(image,
                                     mode='max')

    with asection("max_projection_att"):
        max_projection_att = rgb_project(image,
                                         mode='max',
                                         attenuation=0.05)

    with asection("color_projection"):
        color_projection = rgb_project(image,
                                       mode='color',
                                       attenuation=0.1,
                                       cmap='turbo')

    with asection("color_max_projection"):
        color_max_projection = rgb_project(image,
                                           mode='colormax',
                                           attenuation=0.05,
                                           cmap='turbo',
                                           depth_stabilisation=True)

    with asection("color_max_projection_dg"):
        color_max_projection_dg = rgb_project(image,
                                              mode='colormax',
                                              attenuation=0.05,
                                              depth_gamma=0.6,
                                              cmap='turbo')

    with asection("color_max_projection_bottom"):
        color_max_projection_bottom = rgb_project(image,
                                                  mode='colormax',
                                                  attenuation=0.1,
                                                  cmap='turbo',
                                                  dir=+1)

    image_shifted = sp.ndimage.shift(image, shift=(50, 70, -23), order=1, mode='nearest')

    with asection("color_max_projection_stabilised"):
        color_max_projection_stabilised = rgb_project(image_shifted,
                                                      mode='colormax',
                                                      attenuation=0.05,
                                                      cmap='turbo',
                                                      depth_stabilisation=True)

    # projection = Backend.to_numpy(color_max_projection)
    # png_image = Image.fromarray(projection)
    # png_image.save('test.png')

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(max_projection), name='max_projection', rgb=True)
        viewer.add_image(_c(max_projection_att), name='max_projection_att', rgb=True)
        viewer.add_image(_c(color_projection), name='color_projection', rgb=True)
        viewer.add_image(_c(color_max_projection), name='color_max_projection', rgb=True)
        viewer.add_image(_c(color_max_projection_stabilised), name='color_max_projection_stabilised', rgb=True)
        viewer.add_image(_c(color_max_projection_dg), name='color_max_projection_dg', rgb=True)
        viewer.add_image(_c(color_max_projection_bottom), name='color_max_projection_bottom', rgb=True)


if __name__ == "__main__":
    demo_projection_cupy()
    demo_projection_numpy()
