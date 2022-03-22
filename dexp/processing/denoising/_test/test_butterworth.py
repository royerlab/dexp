from dexp.processing.denoising.demo.demo_2D_butterworth import _demo_butterworth
from dexp.utils.testing.testing import execute_both_backends


@execute_both_backends
def test_butterworth():
    assert _demo_butterworth(display=False) >= 0.608 - 0.03
