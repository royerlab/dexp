# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
import time

import numpy
from napari import gui_qt, Viewer
from tifffile import imread

from dexp.optics.psf.simple_microscope_psf import SimpleMicroscopePSF
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.multiview_lightsheet.simview_fusion import simview_fuse_2I2D
from dexp.processing.utils.scatter_gather import scatter_gather
from dexp.utils.timeit import timeit

filepath = '/home/royer/Desktop/test_data/embryo_4views.tif'


def demo_simview_fuse_numpy():
    backend = NumpyBackend()
    simview_fuse(backend)


def demo_simview_fuse_cupy():
    try:
        backend = CupyBackend(enable_memory_pool=False)
        simview_fuse(backend)
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def simview_fuse(backend):
    start = time.time()

    print(f"Loading data...")
    array = imread(filepath)
    print(f"Done loading.")

    C0L0 = array[0]
    C0L1 = array[1]
    C1L0 = array[2]
    C1L1 = array[3]

    # we assume that the stacks have the same and correct relative orientations:
    C1L0 = numpy.flip(C1L0, -1)
    C1L1 = numpy.flip(C1L1, -1)

    CxLx, shifts = simview_fuse_2I2D(backend, C0L0, C0L1, C1L0, C1L1)
    CxLx = CxLx.astype(numpy.float32)
    print(f"Shifts = {shifts}")

    stop = time.time()
    print(f"Elapsed fusion time:  {stop - start} (includes loading)")


    CxLx = backend.to_numpy(CxLx)

    psf = SimpleMicroscopePSF()
    psf_kernel = psf.generate_xyz_psf(dxy=0.485,
                                      dz=4 * 0.485,
                                      xy_size=17,
                                      z_size=31)
    psf_kernel /= psf_kernel.sum()

    def f(image):
        return lucy_richardson_deconvolution(backend,
                                             image=image,
                                             psf=psf_kernel,
                                             num_iterations=15,
                                             max_correction=2,
                                             power=1.5)

    with timeit("lucy_richardson_deconvolution"):
        CxLx_deconvolved = scatter_gather(backend, f, CxLx, chunks=512, margins=31//2, to_numpy=True)


    with gui_qt():
        def _c(array):
            return backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1500), scale=(4, 1, 1), blending='additive')
        viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive')
        viewer.add_image(_c(C1L0), name='C1L0', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive')
        viewer.add_image(_c(C1L1), name='C1L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive')
        viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', colormap='viridis')
        viewer.add_image(_c(CxLx_deconvolved), name='CxLx_deconvolved', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', colormap='viridis')


# demo_simview_fuse_numpy()
demo_simview_fuse_cupy()
