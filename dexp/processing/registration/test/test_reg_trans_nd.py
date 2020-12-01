from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.registration.reg_trans_nd import register_translation_nd


def test_register_translation_nd_numpy():
    backend = NumpyBackend()
    _register_translation_nd(backend)


def test_register_translation_nd_cupy():
    try:
        backend = CupyBackend()
        _register_translation_nd(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _register_translation_nd(backend):
    xp = backend.get_xp_module()
    sp = backend.get_sp_module()

    image1 = xp.random.uniform(0, 1, size=(257, 237))
    image2 = xp.random.uniform(0, 1, size=(257, 237))

    model = register_translation_nd(backend, image1, image2)
    print(model)

    assert model.confidence < 0.3

    image1 *= 0
    image2 *= 0

    model = register_translation_nd(backend, image1, image2)
    print(model)

    assert model.confidence < 0.3
