from numpy.linalg import norm
from skimage.util import random_noise

from aydin.corrections.broken_pixels.broken_pixels import BrokenPixelsCorrection
from aydin.io.datasets import camera, normalise
from aydin.util.log.log import Log


def test_suppress_fixed_background_real():
    Log.override_test_exclusion = True

    image = normalise(camera())
    noisy = random_noise(image, mode="s&p", amount=0.03, seed=0, clip=False)

    bpc = BrokenPixelsCorrection(lipschitz=0.15)

    corrected = bpc.correct(noisy)

    error0 = norm(noisy - image, ord=1) / image.size
    error = norm(corrected - image, ord=1) / image.size

    print(f"Error noisy = {error0}")
    print(f"Error = {error}")

    assert error < 0.001
    assert error0 > 4 * error
