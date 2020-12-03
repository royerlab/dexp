# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff

import numpy
from napari import gui_qt, Viewer
from tifffile import imread

from dexp.processing.backends.backend import Backend
from dexp.processing.backends.cupy_backend import CupyBackend
from dexp.processing.backends.numpy_backend import NumpyBackend
from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L
from dexp.utils.timeit import timeit

filepath = '/home/royer/Desktop/test_data/embryo_4views.tif'


def demo_simview_fusion_numpy():
    with NumpyBackend():
        simview_fusion()


def demo_simview_fusion_cupy():
    try:
        with CupyBackend():
            simview_fusion()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def simview_fusion():
    with timeit("imread"):
        array = imread(filepath)

    C0L0 = array[0]
    C0L1 = array[1]
    C1L0 = array[2]
    C1L1 = array[3]

    # we assume that the stacks have the same and correct relative orientations:
    C1L0 = numpy.flip(C1L0, -1)
    C1L1 = numpy.flip(C1L1, -1)

    with timeit("simview_fuse_2I2D"):
        CxLx, model = simview_fuse_2C2L(C0L0, C0L1, C1L0, C1L1)
    # CxLx = CxLx.astype(numpy.float32)
    print(f"Model = {model}")

    with timeit("to_numpy"):
        CxLx = Backend.to_numpy(CxLx)
    #
    # psf = SimpleMicroscopePSF()
    # psf_kernel = psf.generate_xyz_psf(dxy=0.485,
    #                                   dz=4 * 0.485,
    #                                   xy_size=17,
    #                                   z_size=17)
    # psf_kernel /= psf_kernel.sum()
    #
    # def f(image):
    #     return lucy_richardson_deconvolution(backend,
    #                                          image=image,
    #                                          psf=psf_kernel,
    #                                          num_iterations=15,
    #                                          max_correction=2,
    #                                          power=1.5,
    #                                          blind_spot=3)
    #
    # with timeit("lucy_richardson_deconvolution"):
    #     CxLx_deconvolved = scatter_gather(backend, f, CxLx, chunks=512, margins=17//2, to_numpy=True)

    with gui_qt():
        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(C0L0), name='C0L0', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C0L1), name='C0L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C1L0), name='C1L0', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(C1L1), name='C1L1', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', visible=False)
        viewer.add_image(_c(CxLx), name='CxLx', contrast_limits=(0, 1200), scale=(4, 1, 1), blending='additive', colormap='viridis')
        # viewer.add_image(_c(CxLx_deconvolved), name='CxLx_deconvolved', contrast_limits=(0, 1000), scale=(4, 1, 1), blending='additive', colormap='viridis')


if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_simview_fusion_cupy()
