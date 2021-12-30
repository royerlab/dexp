from dexp.processing.backends import Backend, CupyBackend
from dexp.processing.registration.demo.demo_warp_ms_3d import _register_warp_3d_ms

# TODO: implement numpy version of warp.
# def test_register_warp_nD_numpy():
#     backend = NumpyBackend()
#     register_warp_nD(backend)


def test_register_warp_ms_3d_cupy():
    try:
        with CupyBackend():
            register_warp_ms_3d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_warp_ms_3d(length_xy=128, warp_grid_size=3):
    xp = Backend.get_xp_module()

    image, warped, unwarped, model = _register_warp_3d_ms(
        length_xy=length_xy, warp_grid_size=warp_grid_size, display=False
    )

    error_warped = xp.mean(xp.absolute(image - warped))
    error_unwarped = xp.mean(xp.absolute(image - unwarped))
    print(f"error_warped = {error_warped}")
    print(f"error_unwarped = {error_unwarped}")

    assert error_unwarped < error_warped
    assert error_unwarped < 33
