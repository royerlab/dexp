from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.functional.test.test_reg_trans_2d_numpy import register_translation_2d


def test_register_translation_2d_numpy():
    try:
        backend = CupyBackend()
        register_translation_2d(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


