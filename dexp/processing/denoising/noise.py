from typing import Optional

from dexp.utils import xpArray
from dexp.utils.backends import Backend


def add_noise(
    image: xpArray,
    normalise: bool = False,
    intensity: int = 5,
    variance: float = 0.01,
    seed: Optional[int] = None,
):
    """
    Adds Poisson and Gaussian noise to an image.

    Parameters
    ----------
    image: array
        Image to add noise to.
    intensity: int
        intensity for Possion noise
    variance: float
        Gaussian noise variance.
    clip
    seed

    Returns
    -------

    """

    # Backend:
    xp = Backend.get_xp_module(image)

    # copy image:
    image = image.copy()

    # Normalise:
    if normalise:
        image -= image.min()
        image /= image.max()

    # Sets teh seed:
    if seed is not None:
        seed = abs(seed)
        xp.random.seed(seed)

    # Applies Poisson noise:
    if intensity is not None:
        image = xp.random.poisson(image * intensity) / intensity

    # Applies Gaussian noise:
    image += xp.random.normal(0, variance, size=image.shape)

    # Resets dtype:
    image = image.astype(xp.float32, copy=False)

    # Returns image:
    return image
