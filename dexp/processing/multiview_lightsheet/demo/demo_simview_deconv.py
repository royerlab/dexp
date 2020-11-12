# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
import time

from napari import gui_qt, Viewer
from tifffile import imread

from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.restoration.clean_dark_regions import clean_dark_regions
from dexp.processing.restoration.dehazing import dehaze
from dexp.processing.utils.scatter_gather import scatter_gather
from dexp.utils.timeit import timeit

filepath = '/home/royer/Desktop/test_data/embryo_4views.tif'


def demo_simview_deconv_numpy():
    backend = NumpyBackend()
    simview_deconv(backend)


def demo_simview_deconv_cupy():
    try:
        backend = CupyBackend(enable_memory_pool=False)
        simview_deconv(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def simview_deconv(backend):
    start = time.time()

    print(f"Loading data...")
    array = imread(filepath)
    print(f"Done loading.")

    view = array[0]

    with timeit(f"Dehaze view ..."):
        view_dehazed = dehaze(backend, view, size=65, minimal_zero_level=0)

    with timeit(f"Denoise dark regions of CxLx..."):
        dark_denoise_threshold: int = 80
        dark_denoise_size: int = 9
        view_dehazed_darkdenoised = clean_dark_regions(backend,
                                                       view_dehazed,
                                                       size=dark_denoise_size,
                                                       threshold=dark_denoise_threshold)

    psf = nikon16x08na()

    def f(image):
        return lucy_richardson_deconvolution(backend,
                                             image=image,
                                             psf=psf_kernel,
                                             num_iterations=15,
                                             max_correction=2,
                                             power=2,
                                             blind_spot=3)

    with timeit("lucy_richardson_deconvolution"):
        view_dehazed_darkdenoised_deconvolved = scatter_gather(backend, f, view_dehazed_darkdenoised, chunks=512, margins=17, to_numpy=True)

    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(view), name='view', contrast_limits=(0, 2000), scale=(4, 1, 1))
        viewer.add_image(_c(view_dehazed), name='view_dehazed', contrast_limits=(0, 2000), scale=(4, 1, 1))
        viewer.add_image(_c(view_dehazed_darkdenoised), name='view_dehazed_darkdenoised', contrast_limits=(0, 2000), scale=(4, 1, 1))
        viewer.add_image(_c(view_dehazed_darkdenoised_deconvolved), name='view_deconvolved', contrast_limits=(0, 2000), scale=(4, 1, 1))


# demo_simview_fuse_numpy()
demo_simview_deconv_cupy()
