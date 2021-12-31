import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.interpolation.warp import warp
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_warp_3d_numpy():
    try:
        with NumpyBackend():
            _demo_warp_3d()
    except NotImplementedError:
        print("Numpy version not yet implemented")


def demo_warp_3d_cupy():
    try:
        with CupyBackend():
            _demo_warp_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_warp_3d(length_xy=256, grid_size=8):
    with timeit("generate data"):
        _, _, image = generate_nuclei_background_data(
            add_noise=True, length_xy=length_xy, length_z_factor=1, zoom=2, dtype=numpy.float32
        )

    newimage = image[0:512, 0:511, 0:509]
    image = newimage

    print(f"shape={image.shape}")

    magnitude = 15
    vector_field = numpy.random.uniform(low=-magnitude, high=+magnitude, size=(grid_size,) * 3 + (3,))

    with timeit("warp"):
        warped = warp(image, vector_field, vector_field_upsampling=4, image_to_backend=True)

    with timeit("dewarped"):
        dewarped = warp(warped, -vector_field, vector_field_upsampling=4, image_to_backend=True)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(
            _c(image), name="image", colormap="bop orange", blending="additive", rendering="attenuated_mip"
        )
        viewer.add_image(
            _c(warped), name="warped", colormap="bop purple", blending="additive", rendering="attenuated_mip"
        )
        viewer.add_image(
            _c(dewarped), name="dewarped", colormap="bop blue", blending="additive", rendering="attenuated_mip"
        )
        viewer.camera.ndisplay = 3


if __name__ == "__main__":
    demo_warp_3d_cupy()
    # demo_warp_3d_numpy()
