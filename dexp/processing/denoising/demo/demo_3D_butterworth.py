# flake8: noqa
from arbol import Arbol

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.processing.denoising.butterworth import calibrate_denoise_butterworth
from dexp.processing.denoising.metrics import psnr, ssim
from dexp.processing.denoising.noise import add_noise
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_butterworth_numpy():
    with NumpyBackend():
        _demo_butterworth()


def demo_butterworth_cupy():
    try:
        with CupyBackend():
            _demo_butterworth()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_butterworth(display=True):
    """
    Demo for self-supervised denoising using camera image with synthetic noise
    """
    # Backend:
    xp = Backend.get_xp_module()

    Arbol.enable_output = True
    Arbol.set_log_max_depth(5)

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(add_noise=True, length_xy=320, length_z_factor=1)

    image = Backend.to_backend(image, dtype=xp.float32)
    image /= image.max()

    noisy = add_noise(image)

    function, parameters = calibrate_denoise_butterworth(noisy, mode="xy-z")
    denoised = function(noisy, **parameters)

    image = xp.clip(image, 0, 1)
    noisy = xp.clip(noisy, 0, 1)
    denoised = xp.clip(denoised, 0, 1)
    psnr_noisy = psnr(image, noisy)
    ssim_noisy = ssim(image, noisy)
    psnr_denoised = psnr(image, denoised)
    ssim_denoised = ssim(image, denoised)
    print("        noisy   :", psnr_noisy, ssim_noisy)
    print("lowpass denoised:", psnr_denoised, ssim_denoised)

    if display:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(Backend.to_numpy(image), name="image")
        viewer.add_image(Backend.to_numpy(noisy), name="noisy")
        viewer.add_image(Backend.to_numpy(denoised), name="denoised")
        napari.run()

    return ssim_denoised


if __name__ == "__main__":
    demo_butterworth_cupy()
    demo_butterworth_numpy()
