
import scipy
from pytest import approx
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.functional.reg_trans_2d import register_translation_2d_skimage


def test_register_translation_2d_numpy():
    backend = NumpyBackend()
    register_translation_2d(backend)

def test_register_translation_2d_cupy():
    try:
        backend = CupyBackend()
        register_translation_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_translation_2d(backend):

    image = binary_blobs(length=100, n_dim=2, blob_size_fraction=0.04, volume_fraction=0.01).astype('f4')
    image = gaussian(image, sigma=1)
    translated_image = scipy.ndimage.shift(image, shift=(13, -5))

    image = random_noise(image, mode='speckle', var=0.5)
    translated_image = random_noise(translated_image, mode='speckle', var=0.5)

    shifts, error = register_translation_2d_skimage(backend, image, translated_image).get_shift_and_error()


    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='array_first')
    #     viewer.add_image(translated_image, name='array_second')

    print(shifts, error)

    assert shifts[0] == approx(-13, abs=0.5)
    assert shifts[1] == approx(5, abs=0.5)


