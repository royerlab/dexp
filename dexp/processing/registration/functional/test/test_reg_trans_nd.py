import scipy
from pytest import approx
from skimage.data import binary_blobs
from skimage.filters import gaussian
from skimage.util import random_noise

from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.functional.reg_trans_nd import register_translation_nd


def test_register_translation_nD_numpy():
    backend = NumpyBackend()
    register_translation_nD(backend)

def test_register_translation_nD_cupy():
    try:
        backend = CupyBackend()
        register_translation_nD(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")

def register_translation_nD(backend):

    image = binary_blobs(length=100, n_dim=3, blob_size_fraction=0.04, volume_fraction=0.01).astype('f4')
    image = gaussian(image, sigma=1)
    translated_image = scipy.ndimage.shift(image, shift=(1, 5, -13))

    image = random_noise(image, mode='speckle', var=0.5)
    translated_image = random_noise(translated_image, mode='speckle', var=0.5)

    shifts, error = register_translation_nd(backend, image, translated_image)

    print(shifts, error)

    # from napari import Viewer, gui_qt
    # with gui_qt():
    #     viewer = Viewer()
    #     viewer.add_image(image, name='array_first')
    #     viewer.add_image(translated_image, name='array_second')

    assert shifts[0] == approx(-1, abs=0.2)
    assert shifts[1] == approx(-5, abs=0.2)
    assert shifts[2] == approx(13, abs=0.2)

