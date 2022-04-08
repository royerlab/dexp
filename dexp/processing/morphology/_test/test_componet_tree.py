import numpy as np
import pytest
from scipy.ndimage import label
from skimage.measure import regionprops

from dexp.processing.morphology import area_white_top_hat
from dexp.utils.backends.backend import Backend
from dexp.utils.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "dexp_nuclei_background_data",
    [dict(length_xy=128, length_z_factor=1, dtype=np.uint16, add_noise=False, background_strength=0.5)],
    indirect=True,
)
def test_white_area_top_hat(dexp_nuclei_background_data, display_test: bool):
    """Test white area top hat by removing background from synthetic nuclei data.
    This operation keeps every object below the area threshold estimated from the ground-truth cells data.
    """
    cells, background, both = dexp_nuclei_background_data
    sampling = 4

    labels, _ = label(Backend.to_numpy(cells > 0.5))
    max_area = 0
    for props in regionprops(labels):
        max_area = max(max_area, props.area)

    max_area = max_area / (sampling ** 3) + 1

    estimated_cells = area_white_top_hat(both, area_threshold=max_area, sampling=sampling)

    if display_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(estimated_cells), name="White Top Hat")
        viewer.add_image(Backend.to_numpy(both), name="Input (Both)")
        viewer.add_image(Backend.to_numpy(cells), name="Cells")
        viewer.add_image(Backend.to_numpy(background), name="Background")
        viewer.grid.enabled = True

        napari.run()

    xp = Backend.get_xp_module(both)
    estimated_cells = estimated_cells / estimated_cells.max()  # cells are binary
    error = xp.abs(estimated_cells - cells).mean()
    print(f"Error = {error}")
    assert error < 1e-1


if __name__ == "__main__":
    from dexp.utils.testing import test_as_demo

    # the same as executing from the CLI
    # pytest <file name> -s --display True
    test_as_demo(__file__)
