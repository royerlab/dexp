from dexp.processing.registration.demo.demo_warp_ms_2d import _register_warp_2d_ms
from dexp.utils.backends import Backend, CupyBackend

# TODO: implement numpy version of warp.
# def test_register_warp_nD_numpy():
#     backend = NumpyBackend()
#     register_warp_nD(backend)


def test_register_warp_ms_2d_cupy():
    try:
        with CupyBackend():
            register_warp_ms_2d()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def register_warp_ms_2d(warp_grid_size=3, reg_grid_size=6):
    xp = Backend.get_xp_module()

    image, warped, unwarped, model = _register_warp_2d_ms(
        warp_grid_size=warp_grid_size, reg_grid_size=reg_grid_size, display=False
    )

    error_warped = xp.mean(xp.absolute(image - warped))
    error_unwarped = xp.mean(xp.absolute(image - unwarped))
    print(f"error_warped = {error_warped}")
    print(f"error_unwarped = {error_unwarped}")

    assert error_unwarped < error_warped
    assert error_unwarped < 19
