from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.translation_nd import register_translation_nd


def test_register_translation_nd_numpy():
    with NumpyBackend():
        _register_translation_nd()


def test_register_translation_nd_cupy():
    try:
        with CupyBackend():
            _register_translation_nd()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _register_translation_nd():
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image1 = xp.random.uniform(0, 1, size=(257, 237))
    image2 = xp.random.uniform(0, 1, size=(257, 237))

    model = register_translation_nd(image1, image2)
    print(model)

    assert model.confidence < 0.3

    image1 *= 0
    image2 *= 0

    model = register_translation_nd(image1, image2)
    print(model)

    assert model.confidence < 0.3
