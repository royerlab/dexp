from dexp.processing.denoising.demo.demo_noise import _demo_noise
from dexp.utils.testing import execute_both_backends


@execute_both_backends
def test_noise():
    _demo_noise(display=False)
