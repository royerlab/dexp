import numpy
from arbol import aprint, asection
from skimage.data import camera

from dexp.processing.registration.translation_nd import register_translation_nd
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_register_translation_2d_numpy():
    with NumpyBackend():
        _register_translation_2d()


def demo_register_translation_2d_cupy():
    try:
        with CupyBackend():
            _register_translation_2d()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def _register_translation_2d(shift=(13, -5), display=True):
    sp = Backend.get_sp_module()

    with asection("generate dataset"):
        image = camera().astype(numpy.float32) / 255
        image = Backend.to_backend(image)
        image = image[0:511, 0:501]

    with asection("shift"):
        shifted = sp.ndimage.shift(image, shift=shift)
        aprint(f"shift applied: {shift}")

    with asection("register_translation_2d"):
        model = register_translation_nd(image, shifted)
        aprint(f"model: {model}")

    with asection("shift back"):
        _, unshifted = model.apply_pair(image, shifted)
        image1_reg, image2_reg = model.apply_pair(image, shifted, pad=False)
        image1_reg_pad, image2_reg_pad = model.apply_pair(image, shifted, pad=True)

    if display:
        from napari import Viewer, gui_qt

        with gui_qt():

            def _c(array):
                return Backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name="image", colormap="bop orange", blending="additive", visible=False)
            viewer.add_image(_c(shifted), name="shifted", colormap="bop blue", blending="additive", visible=False)
            viewer.add_image(_c(unshifted), name="unshifted", colormap="bop blue", blending="additive", visible=False)
            viewer.add_image(
                _c(image1_reg), name="image1_reg", colormap="bop orange", blending="additive", visible=False
            )
            viewer.add_image(_c(image2_reg), name="image2_reg", colormap="bop blue", blending="additive", visible=False)
            viewer.add_image(_c(image1_reg_pad), name="image1_reg_pad", colormap="bop orange", blending="additive")
            viewer.add_image(_c(image2_reg_pad), name="image2_reg_pad", colormap="bop blue", blending="additive")

    return image, shifted, unshifted, model


if __name__ == "__main__":
    demo_register_translation_2d_cupy()
    demo_register_translation_2d_numpy()
