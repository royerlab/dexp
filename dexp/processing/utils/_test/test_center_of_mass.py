import numpy as np
import pytest
from arbol import aprint

from dexp.processing.utils.center_of_mass import center_of_mass
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "dexp_nuclei_background_data",
    [
        dict(
            length_xy=128,
            add_noise=False,
            length_z_factor=1,
            background_strength=0.001,
            sphere=True,
            radius=0.5,
            zoom=2,
            add_offset=False,
            dtype=np.uint16,
        )
    ],
    indirect=True,
)
def test_center_of_mass(dexp_nuclei_background_data, display: bool = False) -> None:
    xp = Backend.get_xp_module()
    sp = Backend.get_sp_module()

    _, _, image = dexp_nuclei_background_data

    com_before = center_of_mass(image)

    shift = xp.array([50, 70, -23])

    image_shifted = sp.ndimage.shift(image, shift=shift, order=1, mode="constant")

    com_after = center_of_mass(image_shifted)

    com_after_bb = center_of_mass(image_shifted, offset_mode="p=75", bounding_box=True)

    err = xp.mean(xp.absolute(com_after - shift - com_before))
    err_bb = xp.mean(xp.absolute(com_after_bb - shift - com_before))

    aprint(f"com_before: {com_before}")
    aprint(f"translation: {shift}")
    aprint(f"com_after: {com_after}")
    aprint(f"com_after_bb: {com_after_bb}")
    aprint(f"Error = {err}")
    aprint(f"Error bounding-box = {err_bb}")

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(image), name="image")
        viewer.add_image(Backend.to_numpy(image_shifted), name="image_shifted")

        napari.run()

    assert err < 10
    assert err_bb < 10
