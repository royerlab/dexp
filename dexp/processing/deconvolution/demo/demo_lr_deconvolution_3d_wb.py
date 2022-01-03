import numpy

from dexp.datasets.synthetic_datasets import generate_nuclei_background_data
from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.filters.fft_convolve import fft_convolve
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit


def demo_lr_deconvolution_numpy():
    with NumpyBackend():
        _demo_lr_deconvolution()


def demo_lr_deconvolution_cupy():
    try:
        with CupyBackend():
            _demo_lr_deconvolution()
    except ModuleNotFoundError:
        print("Cupy module not found! Test passes nevertheless!")


def _demo_lr_deconvolution(length_xy=128):
    xp = Backend.get_xp_module()

    with timeit("generate data"):
        image_gt, background, image = generate_nuclei_background_data(
            add_noise=False, length_xy=length_xy, length_z_factor=1, zoom=2, background_strength=0, add_offset=False
        )
    psf_size = 31

    psf = nikon16x08na(xy_size=psf_size, z_size=psf_size)
    # psf = olympus20x10na()
    psf = psf.astype(dtype=numpy.float16)
    image = image.astype(dtype=numpy.float16)

    blurry = fft_convolve(image, psf)
    blurry = blurry - blurry.min()
    blurry = blurry / blurry.max()
    noisy = blurry
    noisy = noisy + 0.01 * xp.random.uniform(size=blurry.shape)

    max_correction = 8

    iterations = 50

    iterations_wb = 5
    wb_cutoffs = 0.9
    wb_order = 2
    wb_beta = 0.05

    with timeit("deconvolved_wb"):
        deconvolved_wb = lucy_richardson_deconvolution(
            noisy,
            psf,
            num_iterations=iterations_wb,
            back_projection="wb",
            wb_cutoffs=wb_cutoffs,
            wb_beta=wb_beta,
            wb_order=wb_order,
            max_correction=max_correction,
            padding=psf_size,
        )

    with timeit("deconvolved_same"):
        deconvolved_same = lucy_richardson_deconvolution(
            noisy, psf, num_iterations=iterations_wb, max_correction=max_correction, padding=psf_size
        )

    with timeit("deconvolved"):
        deconvolved = lucy_richardson_deconvolution(
            noisy, psf, num_iterations=iterations, max_correction=max_correction, padding=psf_size
        )

    # with timeit("lucy_richardson_deconvolution (scatter-gather)"):
    #     def f(_image):
    #         return lucy_richardson_deconvolution(backend, _image, psf,
    #                                              num_iterations=iterations_wb,
    #                                              back_projection='wb',
    #                                              wb_beta=wb_beta,
    #                                              wb_order=wb_order,
    #                                              max_correction=max_correction,
    #                                              normalise_input=False,
    #                                              padding=16)
    #
    #     deconvolved_sg_wb = scatter_gather_i2i(backend, f, noisy,
    #                                         chunks=length_xy // 2,
    #                                         margins=17,
    #                                         normalise=True,
    #                                         to_numpy=True)

    from napari import Viewer, gui_qt

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(image), name="image")
        viewer.add_image(_c(blurry), name="blurry")
        viewer.add_image(_c(psf), name="psf")
        viewer.add_image(_c(noisy), name="noisy")
        viewer.add_image(_c(deconvolved), name="deconvolved")
        viewer.add_image(_c(deconvolved_same), name="deconvolved_same")
        viewer.add_image(_c(deconvolved_wb), name="deconvolved_wb")
        # viewer.add_image(_c(deconvolved_sg_wb), name='deconvolved_sg_wb')


if __name__ == "__main__":
    demo_lr_deconvolution_cupy()
    demo_lr_deconvolution_numpy()
