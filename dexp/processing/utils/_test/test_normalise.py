import numpy as np
import pytest
from arbol import aprint

from dexp.processing.utils.normalise import Normalise
from dexp.utils.backends import Backend
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
@pytest.mark.parametrize(
    "dexp_nuclei_background_data",
    [dict(length_xy=128, dtype=np.float32)],
    indirect=True,
)
def test_normalise(dexp_nuclei_background_data):
    _, _, image = dexp_nuclei_background_data
    image = image.astype(np.uint16)  # required to convert afterwards

    normalise = Normalise(image, low=-0.5, high=1, in_place=False, clip=True, dtype=np.float32)

    image_normalised = normalise.forward(image)
    image_denormalised = normalise.backward(image_normalised)

    assert image_normalised.dtype == np.float32
    assert image_denormalised.dtype == image.dtype

    assert image_normalised.shape == image.shape
    assert image_denormalised.shape == image.shape

    assert image_normalised.min() >= -0.5
    assert image_normalised.max() <= 1
    assert image_normalised.max() - image_normalised.min() >= 1.5

    assert image_denormalised.min() * (1 + 1e-3) >= image.min()
    assert image_denormalised.max() <= (1 + 1e-3) * image.max()
    assert (image_denormalised.max() - image_denormalised.min()) * (1 + 1e-3) >= image.max() - image.min()

    xp = Backend.get_xp_module()
    error = xp.median(xp.abs(image - image_denormalised)).item()
    aprint(f"Error = {error}")
    assert error < 1e-6
