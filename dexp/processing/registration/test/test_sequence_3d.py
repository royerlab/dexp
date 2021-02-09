from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.registration.demo.demo_sequence_3d import _register_sequence_3d


# TODO: implement numpy version of warp.
# def test_register_sequence_2d_numpy():
#     with  NumpyBackend():
#         register_sequence_2d()


def test_register_translation_3d_cupy():
    try:
        with CupyBackend():
            register_sequence_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_sequence_3d(length_xy=256, n=128):
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    image, shifted, stabilised, model = _register_sequence_3d(length_xy=length_xy,
                                                              n=n,
                                                              display=False)

    assert len(model) == image.shape[0]
