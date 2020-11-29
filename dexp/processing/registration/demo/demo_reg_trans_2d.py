import numpy
from skimage.data import camera

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_nd import register_translation_nd
from dexp.utils.timeit import timeit


def demo_register_translation_2d_numpy():
    backend = NumpyBackend()
    _register_translation_2d(backend)


def demo_register_translation_2d_cupy():
    try:
        backend = CupyBackend()
        _register_translation_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def _register_translation_2d(backend, shift=(13, -5), display=True):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    with timeit("generate dataset"):
        image = camera().astype(numpy.float32) / 255
        image = backend.to_backend(image)
        image = image[0:511, 0:501]

    with timeit("shift"):
        shifted = sp.ndimage.shift(image, shift=shift)
        print(f"shift applied: {shift}")

    with timeit("register_translation_2d"):
        model = register_translation_nd(backend, image, shifted)
        print(f"model: {model}")

    with timeit("shift back"):
        _, unshifted = model.apply(backend, image, shifted)
        image1_reg, image2_reg = model.apply(backend, image, shifted, pad=False)
        image1_reg_pad, image2_reg_pad = model.apply(backend, image, shifted, pad=True)

    if display:
        from napari import Viewer, gui_qt
        with gui_qt():
            def _c(array):
                return backend.to_numpy(array)

            viewer = Viewer()
            viewer.add_image(_c(image), name='image', colormap='bop orange', blending='additive', visible=False)
            viewer.add_image(_c(shifted), name='shifted', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(unshifted), name='unshifted', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(image1_reg), name='image1_reg', colormap='bop orange', blending='additive', visible=False)
            viewer.add_image(_c(image2_reg), name='image2_reg', colormap='bop blue', blending='additive', visible=False)
            viewer.add_image(_c(image1_reg_pad), name='image1_reg_pad', colormap='bop orange', blending='additive')
            viewer.add_image(_c(image2_reg_pad), name='image2_reg_pad', colormap='bop blue', blending='additive')

    return image, shifted, unshifted, model


if __name__ == "__main__":
    demo_register_translation_2d_cupy()
    demo_register_translation_2d_numpy()
