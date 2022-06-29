# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff
from napari import Viewer, gui_qt
from tifffile import imread

from dexp.optics.psf.standard_psfs import nikon16x08na
from dexp.processing.deconvolution.lr_deconvolution import lucy_richardson_deconvolution
from dexp.processing.restoration.dehazing import dehaze
from dexp.processing.utils.scatter_gather_i2i import scatter_gather_i2i
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend
from dexp.utils.timeit import timeit

filepath = "/home/royer/Desktop/test_data/embryo_fused.tif"


def demo_simview_deconv_numpy():
    with NumpyBackend():
        simview_deconv()


def demo_simview_deconv_cupy():
    try:
        with CupyBackend(enable_unified_memory=True):
            simview_deconv()
    except ModuleNotFoundError:
        print("Cupy module not found! demo ignored")


def simview_deconv():
    xp = Backend.get_xp_module()

    print("Loading data...")
    array = imread(filepath)
    print("Done loading.")

    # array = array[150 - 64:150 + 64,
    #         1533 - 256:1533 + 256,
    #         931 - 256:931 + 256]

    view = Backend.to_backend(array, dtype=xp.float32)

    # with timeit(f"Clip view ..."):
    #     view = xp.clip(view, a_min=0, a_max=3048, out=view)

    with timeit("Dehaze view ..."):
        view_dehazed = dehaze(view, size=65, minimal_zero_level=0)

    min_value = view_dehazed.min()
    max_value = view_dehazed.max()

    print(f"min={min_value}, max={max_value}")

    psf_size = 31
    psf = nikon16x08na(xy_size=psf_size, z_size=psf_size)

    parameters = {
        "psf": psf,
        "normalise_input": True,
        "normalise_minmax": (min_value, max_value),
        "max_correction": None,
        "power": 1.0,
        # 'blind_spot': 3,
        # 'blind_spot_mode': 'median+uniform',
        # 'blind_spot_axis_exclusion': (0,),
        "clip_output": False,
    }

    parameters_wb = {
        "back_projection": "wb",
        "wb_cutoffs": 0.9,
        "wb_order": 2,
        "wb_beta": 0.05,
    }

    with timeit("lucy_richardson_deconvolution"):

        def f(_image):
            return lucy_richardson_deconvolution(image=_image, num_iterations=10, **parameters)

        view_dehazed_deconvolved = scatter_gather_i2i(
            view_dehazed, f, tiles=320, margins=psf_size, clip=False, to_numpy=True
        )

    with timeit("lucy_richardson_deconvolution_wb"):

        def f(_image):
            return lucy_richardson_deconvolution(image=_image, num_iterations=3, **parameters, **parameters_wb)

        view_dehazed_deconvolved_wb = scatter_gather_i2i(
            view_dehazed, f, tiles=320, margins=psf_size, clip=False, to_numpy=True
        )

    with gui_qt():

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(_c(view), name="raw", contrast_limits=(0, 2000), scale=(4, 1, 1), visible=False)
        viewer.add_image(
            _c(view_dehazed),
            name="dehazed",
            contrast_limits=(0, 2000),
            scale=(4, 1, 1),
            colormap="bop purple",
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            _c(view_dehazed_deconvolved),
            name="view_dehazed_deconvolved",
            contrast_limits=(0, 13000),
            scale=(4, 1, 1),
            colormap="bop blue",
            blending="additive",
        )
        viewer.add_image(
            _c(view_dehazed_deconvolved_wb),
            name="view_dehazed_deconvolved_wb",
            contrast_limits=(0, 13000),
            scale=(4, 1, 1),
            colormap="bop blue",
            blending="additive",
        )


if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_simview_deconv_cupy()
