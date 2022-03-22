from dexp.processing.denoising.demo.demo_2D_gaussian import _demo_gaussian
from dexp.utils.testing import execute_both_backends


@execute_both_backends
def test_gaussian():
    assert _demo_gaussian(display=False) >= 0.600 - 0.02
