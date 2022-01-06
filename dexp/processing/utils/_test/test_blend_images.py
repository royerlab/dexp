import numpy as np
import pytest

from dexp.processing.utils.blend_images import blend_images
from dexp.utils.backends import Backend
from dexp.utils.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "dexp_fusion_test_data",
    [dict(length_xy=128, add_noise=False)],
    indirect=True,
)
def test_blend(dexp_fusion_test_data, display_test: bool) -> None:
    # TODO: improv this testing, too broad
    #  - error too big?
    image_gt, _, blend_a, _, image1, image2 = dexp_fusion_test_data

    blended = blend_images(image1, image2, blend_a)

    assert blended is not image1
    assert blended is not image2
    assert blended.shape == image1.shape
    assert blended.shape == image2.shape
    assert blended.shape == blend_a.shape

    image_gt = Backend.to_numpy(image_gt)
    blended = Backend.to_numpy(blended)
    error = np.median(np.abs(image_gt - blended))
    print(f"Error = {error}")
    assert error < 23

    if display_test:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image_gt, name="image_gt")
        viewer.add_image(image1, name="image1")
        viewer.add_image(image2, name="image2")
        viewer.add_image(blend_a, name="blend_a")
        viewer.add_image(blended, name="blended")

        napari.run()
