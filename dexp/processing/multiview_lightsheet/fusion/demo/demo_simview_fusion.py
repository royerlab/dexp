# You need to point to a tiff file with 4 views as first dim,
# as produced for example by: dexp tiff -w -s [128:129] dataset.zarr -o /home/royer/Desktop/test_data/test_data.tiff


from arbol import aprint, asection
from napari import Viewer, gui_qt
from tifffile import imread

from dexp.processing.multiview_lightsheet.fusion.simview import simview_fuse_2C2L
from dexp.utils.backends import Backend, CupyBackend, NumpyBackend

filepath = "/home/royer/Desktop/test_data/embryo_4views.tif"


def demo_simview_fusion_numpy():
    with NumpyBackend():
        simview_fusion()


def demo_simview_fusion_cupy():
    try:
        with CupyBackend():
            simview_fusion()
    except ModuleNotFoundError:
        aprint("Cupy module not found! demo ignored")


def simview_fusion():
    with asection("imread"):
        array = imread(filepath)

    C0L0 = array[0]
    C0L1 = array[1]
    C1L0 = array[2]
    C1L1 = array[3]

    with asection("simview_fuse_2I2D"):
        CxLx, model, ratios = simview_fuse_2C2L(C0L0, C0L1, C1L0, C1L1)
    aprint(f"model = {model}, ratios={ratios}")

    with asection("to_numpy"):
        CxLx = Backend.to_numpy(CxLx)

    with gui_qt():
        xp = Backend.get_xp_module()

        def _c(array):
            return Backend.to_numpy(array)

        viewer = Viewer()
        viewer.add_image(
            _c(C0L0), name="C0L0", contrast_limits=(0, 1000), scale=(4, 1, 1), blending="additive", visible=False
        )
        viewer.add_image(
            _c(C0L1), name="C0L1", contrast_limits=(0, 1000), scale=(4, 1, 1), blending="additive", visible=False
        )
        viewer.add_image(
            _c(xp.flip(C1L0, -1)),
            name="C1L0",
            contrast_limits=(0, 1000),
            scale=(4, 1, 1),
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            _c(xp.flip(C1L1, -1)),
            name="C1L1",
            contrast_limits=(0, 1000),
            scale=(4, 1, 1),
            blending="additive",
            visible=False,
        )
        viewer.add_image(
            _c(CxLx), name="CxLx", contrast_limits=(0, 1200), scale=(4, 1, 1), blending="additive", colormap="viridis"
        )


if __name__ == "__main__":
    # demo_simview_fuse_numpy()
    demo_simview_fusion_cupy()
