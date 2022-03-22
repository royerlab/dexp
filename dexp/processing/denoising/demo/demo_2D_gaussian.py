# flake8: noqa

from arbol import Arbol
from skimage import data
from skimage.color import rgb2gray

from dexp.processing.denoising.gaussian import calibrate_denoise_gaussian
from dexp.processing.denoising.metrics import psnr, ssim
from dexp.processing.denoising.noise import add_noise
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend


def demo_gaussian_numpy():
    with NumpyBackend():
        _demo_gaussian()


def demo_gaussian_cupy():
    try:
        with CupyBackend():
            _demo_gaussian()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_gaussian(display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """

    # Backend:
    xp = Backend.get_xp_module()

    Arbol.enable_output = True
    Arbol.set_log_max_depth(5)

    image = data.astronaut()
    image = rgb2gray(image)
    image = Backend.to_backend(image)

    noisy = add_noise(image)

    function, parameters = calibrate_denoise_gaussian(noisy)
    denoised = function(noisy, **parameters)

    image = xp.clip(image, 0, 1)
    noisy = xp.clip(noisy, 0, 1)
    denoised = xp.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("         noisy   :", psnr_noisy, ssim_noisy)
    print("gaussian denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(image), name="image")
        viewer.add_image(Backend.to_numpy(noisy), name="noisy")
        viewer.add_image(Backend.to_numpy(denoised), name="denoised")
        napari.run()

    return ssim_denoised


if __name__ == "__main__":
    demo_gaussian_cupy()
    demo_gaussian_numpy()
