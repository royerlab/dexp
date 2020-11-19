import numpy

from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.processing.synthetic_datasets.nuclei_background_data import generate_nuclei_background_data
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.timeit import timeit


def demo_lr_deconvolution_numpy():
    backend = NumpyBackend()
    _demo_lr_deconvolution(backend)


def demo_lr_deconvolution_cupy():
    try:
        backend = CupyBackend()
        _demo_lr_deconvolution(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_lr_deconvolution(backend, length_xy=128):
    xp = backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(backend,
                                                                      add_noise=False,
                                                                      length_xy=length_xy,
                                                                      length_z_factor=1,
                                                                      background_stength=0,
                                                                      add_offset=False)

    psf = nikon16x08na()
    # psf = olympus20x10na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(backend, image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.1 * xp.random.uniform(size=blurry.shape)

    iterations = 50

    with timeit("deconvolved"):
        deconvolved = lucy_richardson_deconvolution(backend, noisy, psf, num_iterations=iterations, padding=16)

    with timeit("deconvolved_power"):
        deconvolved_power = lucy_richardson_deconvolution(backend, noisy, psf, num_iterations=iterations, padding=16, power=2)

    with timeit("deconvolved_blind_spot"):
        deconvolved_blind_spot = lucy_richardson_deconvolution(backend, noisy, psf, num_iterations=iterations, padding=16, blind_spot=3)

    with timeit("deconvolved_blind_spot_power"):
        deconvolved_blind_spot_power = lucy_richardson_deconvolution(backend, noisy, psf, num_iterations=iterations, padding=16, power=1.2, blind_spot=3)

    def f(image):
        return lucy_richardson_deconvolution(backend, image, psf, num_iterations=iterations, padding=16, power=1.2, blind_spot=3)

    with timeit("lucy_richardson_deconvolution (scatter-gather)"):
        deconvolved_blind_spot_power_sg = scatter_gather_i2i(backend, f, noisy, chunks=length_xy // 2, margins=17, to_numpy=True)

    from napari import Viewer, gui_qt
    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name='image')
        viewer.add_image(_c(blurry), name='blurry')
        viewer.add_image(_c(psf), name='psf')
        viewer.add_image(_c(noisy), name='noisy')
        viewer.add_image(_c(deconvolved), name='deconvolved')
        viewer.add_image(_c(deconvolved_power), name='deconvolved_power')
        viewer.add_image(_c(deconvolved_blind_spot), name='deconvolved_blind_spot')
        viewer.add_image(_c(deconvolved_blind_spot_power), name='deconvolved_blind_spot_power')
        viewer.add_image(_c(deconvolved_blind_spot_power_sg), name='deconvolved_blind_spot_power_sg')


demo_lr_deconvolution_cupy()
demo_lr_deconvolution_numpy()
