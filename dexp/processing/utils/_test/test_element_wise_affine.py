import numpy as np
import pytest

from dexp.processing.utils.element_wise_affine import element_wise_affine
from dexp.utils.backends import Backend
from dexp.utils.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "dexp_fusion_test_data",
    [dict(length_xy=128, add_noise=False)],
    indirect=True,
)
def test_element_wise_affine(dexp_fusion_test_data):
    _, _, _, _, image, _ = dexp_fusion_test_data

    transformed = element_wise_affine(image, 2, 0.3)

    transformed = Backend.to_numpy(transformed)
    image = Backend.to_numpy(image)
    error = np.median(np.abs(image * 2 + 0.3 - transformed))
    print(f"error={error}")
    assert error < 22
