# flake8: noqa
import numpy
from arbol import Arbol
from skimage import data
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dexp.processing.denoising.noise import add_noise
from dexp.utils.backends import CupyBackend, NumpyBackend


def demo_noise_numpy():
    with NumpyBackend():
        _demo_noise()


def demo_noise_cupy():
    try:
        with CupyBackend():
            _demo_noise()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_noise(display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    Arbol.enable_output = True
    Arbol.set_log_max_depth(5)

    image = data.astronaut()
    image = rgb2gray(image)

    noisy = add_noise(image)

    image = numpy.clip(image, 0, 1)
    noisy = numpy.clip(noisy, 0, 1)
    psnr_noisy = psnr(image.astype(noisy.dtype), noisy)
    ssim_noisy = ssim(image, noisy)
    print("         noisy   :", psnr_noisy, ssim_noisy)

    assert psnr_noisy > 12 and psnr_noisy < 13
    assert ssim_noisy > 0.28 and ssim_noisy < 0.29

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(image, name="image")
        viewer.add_image(noisy, name="noisy")
        napari.run()


if __name__ == "__main__":
    demo_noise_cupy()
    demo_noise_numpy()
